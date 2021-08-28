import sys
sys.path.append('../../')
import time
import argparse

from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import dgl
import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score, auc, roc_auc_score, precision_recall_curve
from data import load_data, get_train_valid_pos, get_train_neg, get_valid_neg
from EquivHGAE import EquivHGAE
from src.SparseMatrix import SparseMatrix
from src.DataSchema import DataSchema, SparseMatrixData, Relation

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

    # Don't even worry bout it
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
    data_target = SparseMatrix(indices=indices, values=values,
                               shape=shape)
    data_target = data_target.to(device)
    
    return data_target

#%%
def run_model(args):
    schema, schema_out, data, dl = load_data(args.dataset)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data, in_dims = select_features(data, schema, args.feats_type)
    data = data.to(device)
    indices_identity, indices_transpose = data.calculate_indices()


    res_2hop = defaultdict(float)
    res_random = defaultdict(float)
    total = len(list(dl.links_test['data'].keys()))

    for test_edge_type in dl.links_test['data'].keys():
        target_rel = schema.relations[test_edge_type]
        ent_i = target_rel.entities[0]
        n_i = ent_i.n_instances
        ent_j = target_rel.entities[1]
        n_j = ent_j.n_instances
        embedding_schema = DataSchema(schema.entities,
                              [Relation(0,
                                       [ent_i, ent_i],
                                       is_set=True),
                              Relation(1,
                                       [ent_j, ent_j],
                                       is_set=True
                                       )])
        data_embedding = SparseMatrixData(embedding_schema)
        data_embedding[0] = SparseMatrix(
                indices = torch.arange(n_i, dtype=torch.int64).repeat(2,1),
                values=torch.zeros([n_i, args.embedding_dim]),
                shape=(n_i, n_i, args.embedding_dim),
                is_set=True)
        data_embedding[1] = SparseMatrix(
                indices = torch.arange(n_j, dtype=torch.int64).repeat(2,1),
                values=torch.zeros([n_j, args.embedding_dim]),
                shape=(n_j, n_j, args.embedding_dim),
                is_set=True)
        data_embedding.to(device)
        target_schema = DataSchema(schema.entities, [target_rel])
        data_target = SparseMatrixData(target_schema)
        train_pos, valid_pos = dl.get_train_valid_pos()#edge_types=[test_edge_type])
        train_val_pos = get_train_valid_pos(dl, test_edge_type)
        train_pos_head_full, train_pos_tail_full, \
            valid_pos_head, valid_pos_tail = train_val_pos
        net = EquivHGAE(schema, in_dims,
                        layers = args.layers,
                        embedding_dim=args.embedding_dim,
                        embedding_entities=[ent_i, ent_j],
                        activation=eval('nn.%s()' % args.act_fn),
                        final_activation = nn.Identity(),
                        dropout=args.dropout,
                        norm=args.norm,
                        pool_op=args.pool_op,
                        norm_affine=args.norm_affine,
                        output_relations=target_schema.relations)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)


        if args.wandb_log_run:
            wandb.init(config=args,
                settings=wandb.Settings(start_method='fork'),
                project="EquivariantHGN",
                entity='danieltlevy')
            wandb.watch(net, log='all', log_freq=args.wandb_log_param_freq)
        print(args)
        progress = tqdm(range(args.epoch), desc="Epoch 0", position=0, leave=True)
        # training loop
        net.train()
        #early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path='checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, args.num_layers))
        loss_func = nn.BCELoss()
        for epoch in progress:
          train_neg_head_full, train_neg_tail_full = get_train_neg(dl, test_edge_type)
          train_idx = np.arange(len(train_pos_head_full))
          np.random.shuffle(train_idx)
          batch_size = args.batch_size

          for step, start in enumerate(range(0, len(train_pos_head_full), batch_size)):
            # training
            net.train()
            train_pos_head = train_pos_head_full[train_idx[start:start+batch_size]]
            train_neg_head = train_neg_head_full[train_idx[start:start+batch_size]]
            train_pos_tail = train_pos_tail_full[train_idx[start:start+batch_size]]
            train_neg_tail = train_neg_tail_full[train_idx[start:start+batch_size]]
            data_target[0] =  make_target_matrix(target_rel, 
                                              train_pos_head, train_pos_tail,
                                              train_neg_head, train_neg_tail,
                                              device)

            logits = net(data, indices_identity, indices_transpose,
                         data_target, data_embedding)[test_edge_type].values.squeeze()
            logp = torch.sigmoid(logits)
            labels_train = data_target[0].values[:,0]
            train_loss = loss_func(logp, labels_train)

            # autograd
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()


            progress.set_description(f"Epoch {epoch}")
            progress.set_postfix(loss=train_loss.item())
            wandb_log = {'Train Loss': train_loss.item(), 'epoch':epoch}
    
            # validation
            net.eval()
            if step % args.val_every == 0:

                with torch.no_grad():
                    net.eval()
                    valid_neg_head, valid_neg_tail = get_valid_neg(dl, test_edge_type)
                    data_target[0] = make_target_matrix(target_rel,
                                                         valid_pos_head, valid_pos_tail,
                                                         valid_neg_head, valid_neg_tail,
                                                         device)
    
                    logits = net(data, indices_identity, indices_transpose,
                                 data_target, data_embedding)[test_edge_type].values.squeeze()
                    logp = torch.sigmoid(logits)
                    labels_val = data_target[0].values[:,0]
                    val_loss = loss_func(logp, labels_val)
                    left = data_target[0].indices[0].cpu().numpy()
                    right = data_target[0].indices[1].cpu().numpy()
                    edge_list = np.concatenate([left.reshape((1,-1)), right.reshape((1,-1))], axis=0)
                    wandb_log.update({'val_loss': val_loss.item()})
                    res = dl.evaluate(edge_list, logp.cpu().numpy(), labels_val.cpu().numpy())
                    wandb_log.update(res)
            if args.wandb_log_run:
                wandb.log(wandb_log)
            # early stopping
            #early_stopping(val_loss, net)
            #if early_stopping.early_stop:
            #    print('Early stopping!')
            #    break
          #if early_stopping.early_stop:
          #    print('Early stopping!')
          #    break

        # testing with evaluate_results_nc
        '''
        net.load_state_dict(torch.load('checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, args.num_layers)))
        net.eval()
        test_logits = []
        with torch.no_grad():
            test_neigh, test_label = dl.get_test_neigh()
            test_neigh = test_neigh[test_edge_type]
            test_label = test_label[test_edge_type]
            left = np.array(test_neigh[0])
            right = np.array(test_neigh[1])
            mid = np.zeros(left.shape[0], dtype=np.int32)
            labels = torch.FloatTensor(test_label).to(device)
            logits = net(features_list, e_feat, left, right, mid)
            pred = F.sigmoid(logits).cpu().numpy()
            edge_list = np.concatenate([left.reshape((1,-1)), right.reshape((1,-1))], axis=0)
            labels = labels.cpu().numpy()
            dl.gen_file_for_evaluate(test_neigh, pred, test_edge_type, file_path=f"{args.dataset}_{args.run}.txt")
            res = dl.evaluate(edge_list, pred, labels)
            print(res)
            for k in res:
                res_2hop[k] += res[k]
        with torch.no_grad():
            test_neigh, test_label = dl.get_test_neigh_w_random()
            test_neigh = test_neigh[test_edge_type]
            test_label = test_label[test_edge_type]
            left = np.array(test_neigh[0])
            right = np.array(test_neigh[1])
            mid = np.zeros(left.shape[0], dtype=np.int32)
            labels = torch.FloatTensor(test_label).to(device)
            logits = net(features_list, e_feat, left, right, mid)
            pred = F.sigmoid(logits).cpu().numpy()
            edge_list = np.concatenate([left.reshape((1,-1)), right.reshape((1,-1))], axis=0)
            labels = labels.cpu().numpy()
            res = dl.evaluate(edge_list, pred, labels)
            print(res)
            for k in res:
                res_random[k] += res[k]
        '''
    for k in res_2hop:
        res_2hop[k] /= total
    for k in res_random:
        res_random[k] /= total
    print(res_2hop)
    print(res_random)

#%%
def get_hyperparams(argv):
    ap = argparse.ArgumentParser(allow_abbrev=False, description='EquivHGN for Node Classification')
    ap.add_argument('--feats_type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others);' + 
                        '5 - only term features (zero vec for others).')
    ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
    ap.add_argument('--batch_size', type=int, default=8192)
    ap.add_argument('--patience', type=int, default=30, help='Patience.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--dataset', type=str, default='LastFM')
    ap.add_argument('--checkpoint_path', type=str, default='checkpoint/checkpoint.pt')
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
    ap.add_argument('--norm',  type=int, default=1)
    ap.add_argument('--norm_affine', action='store_true')
    ap.set_defaults(norm_affine=True)
    ap.add_argument('--pool_op', type=str, default='mean')
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
