import sys
sys.path.append('../../')

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import wandb
from tqdm import tqdm
from data_lp import load_data_flat, get_train_valid_pos, get_train_neg, \
    get_valid_neg, get_valid_neg_2hop, get_test_neigh_from_file, gen_file_for_evaluate
from EquivHGAE import EquivLinkPredictor
from src.SparseMatrix import SparseMatrix
from src.DataSchema import SparseMatrixData
import warnings

warnings.filterwarnings("ignore", message="Setting attributes on ParameterDict is not supported.")

#%%
def set_seed(seed):
    random.seed(seed, version=2)
    np.random.seed(random.randint(0, 2**32))
    torch.manual_seed(random.randint(0, 2**32))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_features(data, schema, feats_type):
    '''
    TODO: IMPLEMENT THIS
    '''
    # Select features for nodes
    in_dims = {}
    num_relations = len(schema.relations) - len(schema.entities)

    if feats_type == 0:
        # Keep all node attributes
        pass
    '''
    elif feats_type == 1:
        # Set all non-target node attributes to zero
        for ent_i in schema.entities:
            if ent_i.id != target_ent:
                # 10 dimensions for some reason
                n_dim = 10
                rel_id = num_relations + ent_i.id
                data[rel_id] = SparseMatrix.from_other_sparse_matrix(data[rel_id], n_dim)

    features_list = [mat2tensor(features).to(device) for features in features_list]
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []#[features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2[]
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    '''
    for rel in schema.relations:
        in_dims[rel.id] = data[rel.id].n_channels
    return data, in_dims

def loss_fcn(data_pred, data_true):
    return F.binary_cross_entropy(data_pred, data_true)
    
def make_target_matrix(relation, pos_head, pos_tail, neg_head, neg_tail, device):
    n_pos = pos_head.shape[0]
    pos_indices = np.vstack((pos_head, pos_tail))
    pos_values = np.ones((n_pos, 1))
    n_neg = neg_head.shape[0]
    neg_indices = np.vstack((neg_head, neg_tail))
    neg_values = np.zeros((n_neg, 1))
    indices = torch.LongTensor(np.concatenate((pos_indices, neg_indices), 1))
    values = torch.FloatTensor(np.concatenate((pos_values, neg_values), 0))
    shape = (relation.entities[0].n_instances,
             relation.entities[1].n_instances, 1)
    data_target = SparseMatrix(indices=indices, values=values, shape=shape)
    data_target = data_target.to(device).coalesce_()

    return data_target

def make_flat_target_matrix(full_relation, rel_ids, pos_heads, pos_tails, neg_heads, neg_tails, device):
    full_heads, full_tails = np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    for rel_id in rel_ids:
        full_heads = np.concatenate((full_heads, pos_heads[rel_id]))
        full_heads = np.concatenate((full_heads, neg_heads[rel_id]))
        full_tails = np.concatenate((full_tails, pos_tails[rel_id]))
        full_tails = np.concatenate((full_tails, neg_tails[rel_id]))
    n_rels = len(rel_ids)
    indices = torch.LongTensor(np.vstack((full_heads, full_tails)))
    values = torch.zeros((indices.shape[1], n_rels))
    shape = (full_relation.entities[0].n_instances,
             full_relation.entities[1].n_instances, n_rels)
    full_matrix = SparseMatrix(indices=indices, values=values, shape=shape)
    full_matrix = full_matrix.to(device).coalesce_()
    matrix_out = SparseMatrix.from_other_sparse_matrix(full_matrix, 0)

    for rel_id in rel_ids:
        rel_matrix = make_target_matrix(full_relation,  pos_heads[rel_id],
                                        pos_tails[rel_id], neg_heads[rel_id],
                                        neg_tails[rel_id], device)

        rel_matrix_full = SparseMatrix.from_other_sparse_matrix(full_matrix, 1) + rel_matrix
        matrix_out.values = torch.cat([matrix_out.values, rel_matrix_full.values], 1)
        matrix_out.n_channels += 1
    return matrix_out

def combine_matrices_flat(full_relation, a_pos_heads, a_pos_tails, a_neg_heads, a_neg_tails, ids, b_matrix, device):
    '''
    inputs:
        a_heads: a dict of ID : head indices
        a_tails: a dict of ID : tail indices
        ids: IDs with which to access the indices of A
        b_matrix: a matrix whose indices we want to include in output

    returns:
        out_matrix: matrix with indices & values of A as well as indices of B
        valid_masks: a dict of id:indices that correspond to the indices  for
             each of the relations in A
    '''
    full_heads, full_tails = np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    for rel_id in ids:
        full_heads = np.concatenate((full_heads, a_pos_heads[rel_id]))
        full_heads = np.concatenate((full_heads, a_neg_heads[rel_id]))
        full_tails = np.concatenate((full_tails, a_pos_tails[rel_id]))
        full_tails = np.concatenate((full_tails, a_neg_tails[rel_id]))
    n_rels = len(ids)
    indices = torch.LongTensor(np.vstack((full_heads, full_tails)))
    values = torch.zeros((indices.shape[1], 1))
    shape = (full_relation.entities[0].n_instances,
             full_relation.entities[1].n_instances, 1)
    full_a_matrix = SparseMatrix(indices=indices, values=values, shape=shape)
    full_a_matrix = full_a_matrix.to(device).coalesce_()


    b_idx_matrix = SparseMatrix.from_other_sparse_matrix(b_matrix, 1)
    b_idx_matrix.values += 1

    out_idx_matrix = b_idx_matrix + full_a_matrix
    out_matrix = SparseMatrix.from_other_sparse_matrix(out_idx_matrix, 0)


    for rel_id in ids:
        rel_matrix = make_target_matrix(full_relation,  a_pos_heads[rel_id],
                                        a_pos_tails[rel_id], a_neg_heads[rel_id],
                                        a_neg_tails[rel_id], device)


        rel_full_matrix = SparseMatrix.from_other_sparse_matrix(out_idx_matrix, 1) + rel_matrix
        out_matrix.values = torch.cat([out_matrix.values, rel_full_matrix.values], 1)
        out_matrix.n_channels += 1

        rel_idx_matrix = SparseMatrix.from_other_sparse_matrix(rel_matrix, 1)
        rel_idx_matrix.values += 1
        rel_idx_full_matrix = SparseMatrix.from_other_sparse_matrix(out_idx_matrix, 1) + rel_idx_matrix
        out_idx_matrix.values = torch.cat([out_idx_matrix.values, rel_idx_full_matrix.values], 1)
        out_idx_matrix.n_channels += 1

    masks = {}
    for channel_i, rel_id in enumerate(ids):
        masks[rel_id] = out_idx_matrix.values[:,channel_i+1].nonzero().squeeze()
    return out_matrix, masks

def combine_matrices(matrix_a, matrix_b):
    '''
    Given Matrix A (target) and Matrix B (supplement), would like to get a
    matrix that includes the indices of both matrix A and B, but only the
    values of Matrix A. Additionally, return a mask indicating which indices
    corresponding to Matrix A.
    '''
    # We only want indices from matrix_b, not values
    matrix_b_zero = matrix_b.clone()
    matrix_b_zero.values.zero_()
    matrix_combined = matrix_a + matrix_b_zero

    # To determine indices corresponding to matrix_a, make binary matrix
    matrix_a_ones = matrix_a.clone()
    matrix_a_ones.values.zero_()
    matrix_a_ones.values += 1
    mask_matrix_combined = matrix_a_ones + matrix_b_zero
    matrix_a_mask = mask_matrix_combined.values[:,0].nonzero().squeeze()
    return matrix_combined, matrix_a_mask

def coalesce_matrix(matrix):
    coalesced_matrix = matrix.coalesce(op='mean')
    left = coalesced_matrix.indices[0,:]
    right = coalesced_matrix.indices[1,:]
    labels = coalesced_matrix.values.squeeze()
    return coalesced_matrix, left, right, labels

def make_combined_data(schema, input_data, target_rel_id, target_matrix):
    '''
    given dataset and a single target matrix for predictions, produce new dataset
    with indices combining original dataset with new target matrix's indices
    '''
    combined_data = input_data.clone()
    combined_data[target_rel_id] += target_matrix
    return combined_data

def make_target_matrix_test(relation, left, right, labels, device):
    indices = torch.LongTensor(np.vstack((left, right)))
    values = torch.FloatTensor(labels).unsqueeze(1)
    shape = (relation.entities[0].n_instances,
             relation.entities[1].n_instances, 1)
    return SparseMatrix(indices=indices, values=values, shape=shape).to(device)

#%%
def run_model(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Collect data and schema
    schema, data_original, dl = load_data_flat(args.dataset, use_edge_data=args.use_edge_data)
    data, in_dims = select_features(data_original, schema, args.feats_type)
    data = data.to(device)
    
    # Precompute data indices
    indices_identity, indices_transpose = data.calculate_indices()
    # Get target relations and create data structure for embeddings
    target_rel_ids = dl.links_test['data'].keys()
    num_outputs = len(target_rel_ids)
    flat_rel = schema.relations[0]
    target_ents = schema.entities

    data_embedding = SparseMatrixData.make_entity_embeddings(target_ents,
                                                             args.embedding_dim)
    data_embedding.to(device)

    # Get training and validation positive samples now
    train_pos_heads, train_pos_tails = dict(), dict()
    val_pos_heads, val_pos_tails = dict(), dict()
    for target_rel_id in target_rel_ids:
        train_val_pos = get_train_valid_pos(dl, target_rel_id, flat=True)
        train_pos_heads[target_rel_id], train_pos_tails[target_rel_id], \
            val_pos_heads[target_rel_id], val_pos_tails[target_rel_id] = train_val_pos

    # Create network and optimizer
    net = EquivLinkPredictor(schema, in_dims,
                    layers=args.layers,
                    embedding_dim=args.embedding_dim,
                    embedding_entities=target_ents,
                    output_rel=None,
                    activation=eval('nn.%s()' % args.act_fn),
                    final_activation = nn.Identity(),
                    dropout=args.dropout,
                    pool_op=args.pool_op,
                    norm_affine=args.norm_affine,
                    in_fc_layer=args.in_fc_layer,
                    decode = 'equiv',
                    out_dim = num_outputs)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Set up logging and checkpointing
    if args.wandb_log_run:
        wandb.init(config=args,
            settings=wandb.Settings(start_method='fork'),
            project="EquivariantHGN",
            entity='danieltlevy')
        wandb.watch(net, log='all', log_freq=args.wandb_log_param_freq)
    print(args)
    run_name = args.dataset + '_' + str(args.run)
    if args.wandb_log_run and wandb.run.name is not None:
        run_name = run_name + '_' + str(wandb.run.name)
    if args.checkpoint_path != '':
        checkpoint_path = args.checkpoint_path
    else:
        checkpoint_path = f"checkpoint/checkpoint_{run_name}.pt"
    print("Checkpoint Path: " + checkpoint_path)
    val_metric_best = -1e10

    # training
    loss_func = nn.BCELoss()
    progress = tqdm(range(args.epoch), desc="Epoch 0", position=0, leave=True)
    for epoch in progress:
        net.train()
        # Make target matrix and labels to train on
        # Target is same as input
        data_target = data.clone()
        labels_train = torch.Tensor([]).to(device)

        train_neg_heads, train_neg_tails = dict(), dict()
        for target_rel_id in target_rel_ids:
            train_neg_heads[target_rel_id], train_neg_tails[target_rel_id] = get_train_neg(dl, target_rel_id, flat=True)


        train_matrix = make_flat_target_matrix(flat_rel, target_rel_ids,
                                               train_pos_heads,
                                               train_pos_tails,
                                               train_neg_heads,
                                               train_neg_tails,
                                               device)
        data_target[flat_rel.id] = train_matrix


        # Make prediction
        idx_id_tgt, idx_trans_tgt = data_target.calculate_indices()
        output_data = net(data, indices_identity, indices_transpose,
                   data_embedding, data_target, idx_id_tgt, idx_trans_tgt)
        logits_combined = torch.Tensor([]).to(device)
        for rel_channel, target_rel_id in enumerate(target_rel_ids):
            logits_rel = output_data[flat_rel.id].values[:, rel_channel]
            logits_combined = torch.cat([logits_combined, logits_rel])

            labels_train_rel = train_matrix.values[:,rel_channel]
            labels_train = torch.cat([labels_train, labels_train_rel])

        logp = torch.sigmoid(logits_combined)
        train_loss = loss_func(logp, labels_train)

        # autograd
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Update logging
        progress.set_description(f"Epoch {epoch}")
        progress.set_postfix(loss=train_loss.item())
        wandb_log = {'Train Loss': train_loss.item(), 'epoch':epoch}

        # Evaluate on validation set
        if epoch % args.val_every == 0:
            with torch.no_grad():
                net.eval()
                val_neg_heads, val_neg_tails = dict(), dict()
                for target_rel_id in target_rel_ids:
                    if args.val_neg == '2hop':
                        val_neg_heads[target_rel_id], val_neg_tails[target_rel_id] = get_valid_neg_2hop(dl, target_rel_id)
                    else:
                        val_neg_heads[target_rel_id], val_neg_tails[target_rel_id] = get_valid_neg(dl, target_rel_id)

                val_matrix_combined, val_masks = combine_matrices_flat(flat_rel, val_pos_heads,
                                                    val_pos_tails, val_neg_heads,
                                                    val_neg_tails, target_rel_ids, train_matrix,
                                                    device)
                data_target[flat_rel.id] = val_matrix_combined.clone()


                data_target.zero_()
                idx_id_val, idx_trans_val = data_target.calculate_indices()
                output_data = net(data, indices_identity, indices_transpose,
                           data_embedding, data_target, idx_id_val, idx_trans_val)

                left = torch.Tensor([], dtype=np.int32).to(device)
                right = torch.Tensor([], dtype=np.int32).to(device)
                logits_combined = torch.Tensor([]).to(device)
                labels_val = torch.Tensor([]).to(device)

                for rel_channel, rel_id in enumerate(target_rel_ids):
                    mask = val_masks[rel_id]

                    logits_rel = output_data[flat_rel.id].values[:, rel_channel][mask]
                    logits_combined = torch.cat([logits_combined, logits_rel])

                    left_rel, right_rel = val_matrix_combined.indices[:, mask]
                    left = torch.cat([left, left_rel])
                    right = torch.cat([right, right_rel])
                    labels_val_rel = val_matrix_combined.values[:,rel_channel][mask]
                    labels_val = torch.cat([labels_val, labels_val_rel])

                logp = torch.sigmoid(logits_combined)
                val_loss = loss_func(logp, labels_val).item()

                left = left.cpu().numpy()
                right = right.cpu().numpy()
                edge_list = np.concatenate([left.reshape((1,-1)), right.reshape((1,-1))], axis=0)

                wandb_log.update({'val_loss': val_loss})
                res = dl.evaluate(edge_list, logp.cpu().numpy(), labels_val.cpu().numpy())
                val_roc_auc = res['roc_auc']
                val_mrr = res['MRR']
                wandb_log.update(res)
                print("\nVal Loss: {:.3f} Val ROC AUC: {:.3f} Val MRR: {:.3f}".format(
                    val_loss, val_roc_auc, val_mrr))
                if args.val_metric == 'loss':
                    val_metric = -val_loss
                elif args.val_metric == 'roc_auc':
                    val_metric = val_roc_auc
                elif args.val_metric == 'mrr':
                    val_metric = val_mrr

                if val_metric > val_metric_best:
                    val_metric_best = val_metric
                    print("New best, saving")
                    torch.save({
                        'epoch': epoch,
                        'net_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss.item(),
                        'val_loss': val_loss,
                        'val_roc_auc': val_roc_auc,
                        'val_mrr': val_mrr
                        }, checkpoint_path)
                    if args.wandb_log_run:
                        wandb.summary["val_roc_auc_best"] = val_roc_auc
                        wandb.summary["val_mrr_best"] = val_mrr
                        wandb.summary["val_loss_best"] = val_loss
                        wandb.summary["epoch_best"] = epoch
                        wandb.summary["train_loss_best"] = train_loss.item()
                        wandb.save(checkpoint_path)
        if args.wandb_log_run:
            wandb.log(wandb_log)


    # Evaluate on test set
    if args.evaluate:
        print("Evaluating Target Rel " + str(rel_id))
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net_state_dict'])
        net.eval()

        # Target is same as input
        data_target = data.clone()
        with torch.no_grad():

            test_heads_full = dict()
            test_tails_full = dict()
            for rel_id in target_rel_ids:
                test_heads_full[rel_id], test_tails_full[rel_id], test_labels_full = get_test_neigh_from_file(dl, args.dataset, rel_id)

            test_matrix_combined, test_masks = combine_matrices_flat(flat_rel, test_heads_full,
                                                test_tails_full, test_heads_full,
                                                test_tails_full, target_rel_ids, train_matrix,
                                                device)
            data_target[flat_rel.id] = test_matrix_combined.clone()

            data_target.zero_()
            idx_id_tst, idx_trans_tst = data_target.calculate_indices()
            output_data = net(data, indices_identity, indices_transpose,
                       data_embedding, data_target, idx_id_tst, idx_trans_tst)

            for rel_channel, rel_id in enumerate(target_rel_ids):
                mask = test_masks[rel_id]

                logits = output_data[flat_rel.id].values[:, rel_channel][mask]
                logits_combined = torch.cat([logits_combined, logits_rel])

                left, right = test_matrix_combined.indices[:, mask]
                labels_test = test_matrix_combined.values[:,rel_channel][mask]
                left_full = test_heads_full[rel_id]
                right_full = test_tails_full[rel_id]

                pred = torch.sigmoid(logits)

                left = left.cpu().numpy()
                right = right.cpu().numpy()
                edge_list = np.concatenate([left.reshape((1,-1)), right.reshape((1,-1))], axis=0)
                edge_list_full = np.vstack((left_full, right_full))
                file_path = f"test_out/{run_name}.txt"
                gen_file_for_evaluate(dl, edge_list_full, edge_list, pred, rel_id,
                                         file_path=file_path, flat=True)
#%%
def get_hyperparams(argv):
    ap = argparse.ArgumentParser(allow_abbrev=False, description='EquivHGN for Link Prediction')
    ap.add_argument('--feats_type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others);' + 
                        '5 - only term features (zero vec for others).')
    ap.add_argument('--epoch', type=int, default=10, help='Number of epochs.')
    ap.add_argument('--batch_size', type=int, default=100000)
    ap.add_argument('--patience', type=int, default=30, help='Patience.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training \
                    and testing for N times. Default is 1.')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--dataset', type=str, default='LastFM')
    ap.add_argument('--checkpoint_path', type=str, default='')
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--embedding_dim', type=int, default=64)
    ap.add_argument('--fc_layer', type=int, default=64)
    #ap.add_argument('--fc_layers', type=str, nargs='*', default=[64],
    #                    help='Fully connected layers for target embeddings')
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--act_fn', type=str, default='LeakyReLU')
    ap.add_argument('--in_fc_layer', type=int, default=1)
    ap.add_argument('--optimizer', type=str, default='Adam')
    ap.add_argument('--val_every', type=int, default=5)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--norm_affine', type=int, default=1)
    ap.add_argument('--pool_op', type=str, default='mean')
    ap.add_argument('--use_edge_data',  type=int, default=0)
    ap.add_argument('--save_embeddings', dest='save_embeddings', action='store_true', default=True)
    ap.add_argument('--no_save_embeddings', dest='save_embeddings', action='store_false', default=True)
    ap.set_defaults(save_embeddings=True)
    ap.add_argument('--wandb_log_param_freq', type=int, default=100)
    ap.add_argument('--wandb_log_loss_freq', type=int, default=1)
    ap.add_argument('--wandb_log_run', dest='wandb_log_run', action='store_true',
                        help='Log this run in wandb')
    ap.add_argument('--wandb_no_log_run', dest='wandb_log_run', action='store_false',
                        help='Do not log this run in wandb')
    ap.add_argument('--output', type=str)
    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--evaluate', type=int, default=1)
    ap.add_argument('--decoder', type=str, default='equiv')
    ap.add_argument('--val_neg', type=str, default='2hop')
    ap.add_argument('--val_metric', type=str, default='roc_auc')
    ap.add_argument('--lgnn', action='store_true', default=True)
    ap.set_defaults(wandb_log_run=False)
    args, argv = ap.parse_known_args(argv)
    if args.output == None:
        args.output = args.dataset + '_emb.dat'
    if args.in_fc_layer == 1:
        args.in_fc_layer = True
    else:
        args.in_fc_layer = False
    if args.evaluate == 1:
        args.evaluate = True
    else:
        args.evaluate = False
    if args.norm_affine == 1:
        args.norm_affine = True
    else:
        args.norm_affine = False
    if args.use_edge_data == 1:
        args.use_edge_data = True
    else:
        args.use_edge_data = False
    args.layers = [args.width]*args.depth
    if args.fc_layer == 0:
        args.fc_layers = []
    else:
        args.fc_layers = [args.fc_layer]
    return args

    
if __name__ == '__main__':
    argv = sys.argv[1:]
    args = get_hyperparams(argv)
    set_seed(args.seed)
    #%%
    run_model(args)
