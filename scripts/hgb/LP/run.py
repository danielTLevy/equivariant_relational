import sys
sys.path.append('../../')

import torch
import torch.nn as nn
import numpy as np
import wandb
from tqdm import tqdm
from data_lp import load_data, get_train_valid_pos, get_train_neg, \
    get_valid_neg, get_valid_neg_2hop, get_test_neigh_from_file, gen_file_for_evaluate
from EquivHGAE import EquivLinkPredictor, EquivLinkPredictorShared
from src.DataSchema import DataSchema, SparseMatrixData
from src.utils import count_parameters
from utils import get_hyperparams, set_seed, select_features, make_target_matrix, combine_matrices, coalesce_matrix, make_target_matrix_test
import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterDict is not supported.")

#%%
def run_model(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_equiv = args.decoder == 'equiv'

    # Collect data and schema
    schema, data_original, dl = load_data(args.dataset,
                                          use_edge_data=args.use_edge_data,
                                          use_other_edges=args.use_other_edges,
                                          use_node_attrs=args.use_node_attrs,
                                          node_val=args.node_val)
    data, in_dims = select_features(data_original, schema, args.feats_type)
    data = data.to(device)
    
    # Precompute data indices
    indices_identity, indices_transpose = data.calculate_indices()
    # Get target relations and create data structure for embeddings
    target_rel_ids = dl.links_test['data'].keys()
    target_rels = [schema.relations[rel_id] for rel_id in target_rel_ids]
    target_ents = schema.entities
    # Get relations used by decoder
    if use_equiv:
        output_rels = schema.relations
    else:
        output_rels = {rel.id: rel for rel in target_rels}
    data_embedding = SparseMatrixData.make_entity_embeddings(target_ents,
                                                             args.embedding_dim)
    data_embedding.to(device)

    # Get training and validation positive samples now
    train_pos_heads, train_pos_tails = dict(), dict()
    val_pos_heads, val_pos_tails = dict(), dict()
    for target_rel_id in target_rel_ids:
        train_val_pos = get_train_valid_pos(dl, target_rel_id)
        train_pos_heads[target_rel_id], train_pos_tails[target_rel_id], \
            val_pos_heads[target_rel_id], val_pos_tails[target_rel_id] = train_val_pos

    # Get additional indices to be used when making predictions
    pred_idx_matrices = {}
    for target_rel in target_rels:
        if args.pred_indices == 'train':
            train_neg_head, train_neg_tail = get_train_neg(dl, target_rel.id,
                                                       tail_weighted=args.tail_weighted)
            pred_idx_matrices[target_rel.id] = make_target_matrix(target_rel,
                                              train_pos_heads[target_rel.id],
                                              train_pos_tails[target_rel.id],
                                              train_neg_head, train_neg_tail,
                                              device)
        elif args.pred_indices == 'train_neg':
            # Get negative samples twice
            train_neg_head1, train_neg_tail1 = get_train_neg(dl, target_rel.id,
                                                   tail_weighted=args.tail_weighted)
            train_neg_head2, train_neg_tail2 = get_train_neg(dl, target_rel.id,
                                                   tail_weighted=args.tail_weighted)
            pred_idx_matrices[target_rel.id] = make_target_matrix(target_rel,
                                              train_neg_head1, train_neg_tail1,
                                              train_neg_head2, train_neg_tail2,
                                              device)
        elif args.pred_indices == 'none':
            pred_idx_matrices[target_rel.id] = None

    # Create network and optimizer
    if args.sharing:
        net_type = EquivLinkPredictorShared
    else:
        net_type = EquivLinkPredictor
    net = net_type(schema, in_dims,
                    layers=args.layers,
                    embedding_dim=args.embedding_dim,
                    embedding_entities=target_ents,
                    output_rels=output_rels,
                    activation=eval('nn.%s()' % args.act_fn),
                    final_activation = nn.Identity(),
                    dropout=args.dropout,
                    pool_op=args.pool_op,
                    norm_affine=args.norm_affine,
                    norm_embed=args.norm_embed,
                    in_fc_layer=args.in_fc_layer,
                    decode = args.decoder)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Set up logging and checkpointing
    if args.wandb_log_run:
        wandb.init(config=args,
            settings=wandb.Settings(start_method='fork'),
            project="EquivariantHGN_LP",
            entity='danieltlevy')
        wandb.watch(net, log='all', log_freq=args.wandb_log_param_freq)
    print(args)
    print("Number of parameters: {}".format(count_parameters(net)))
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
        if use_equiv:
            # Target is same as input
            target_schema = schema
            data_target = data.clone()
        else:
            # Target is just target relation
            target_schema = DataSchema(schema.entities, target_rels)
            data_target = SparseMatrixData(target_schema)
        labels_train = torch.Tensor([]).to(device)
        for target_rel in target_rels:
            train_neg_head, train_neg_tail = get_train_neg(dl, target_rel.id,
                                                           tail_weighted=args.tail_weighted)
            train_matrix = make_target_matrix(target_rel,
                                              train_pos_heads[target_rel.id],
                                              train_pos_tails[target_rel.id],
                                              train_neg_head, train_neg_tail,
                                              device)
            data_target[target_rel.id] = train_matrix
            labels_train_rel = train_matrix.values.squeeze()
            labels_train = torch.cat([labels_train, labels_train_rel])

        # Make prediction
        if use_equiv:
            idx_id_tgt, idx_trans_tgt = data_target.calculate_indices()
            output_data = net(data, indices_identity, indices_transpose,
                       data_embedding, data_target, idx_id_tgt, idx_trans_tgt)
        else:
            output_data = net(data, indices_identity, indices_transpose,
                         data_embedding, data_target)
        logits_combined = torch.Tensor([]).to(device)
        for target_rel in target_rels:
            logits_rel = output_data[target_rel.id].values.squeeze()
            logits_combined = torch.cat([logits_combined, logits_rel])

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
        net.eval()
        if epoch % args.val_every == 0:
            with torch.no_grad():
                net.eval()
                left = torch.Tensor([]).to(device)
                right = torch.Tensor([]).to(device)
                labels_val = torch.Tensor([]).to(device)
                valid_masks = {}
                for target_rel in target_rels:
                    if args.val_neg == '2hop':
                        valid_neg_head, valid_neg_tail = get_valid_neg_2hop(dl, target_rel.id)
                    elif args.val_neg == 'randomtw':
                        valid_neg_head, valid_neg_tail = get_valid_neg(dl, target_rel.id, tail_weighted=True)
                    else:
                        valid_neg_head, valid_neg_tail = get_valid_neg(dl, target_rel.id)
                    valid_matrix_full = make_target_matrix(target_rel,
                                                     val_pos_heads[target_rel.id], val_pos_tails[target_rel.id],
                                                     valid_neg_head, valid_neg_tail,
                                                     device)
                    valid_matrix, left_rel, right_rel, labels_val_rel = coalesce_matrix(valid_matrix_full)
                    left = torch.cat([left, left_rel])
                    right = torch.cat([right, right_rel])
                    labels_val = torch.cat([labels_val, labels_val_rel])
                    if use_equiv:
                        # Add in additional prediction indices
                        pred_idx_matrix = pred_idx_matrices[target_rel.id]
                        if pred_idx_matrix is None:
                            valid_combined_matrix = valid_matrix
                            valid_mask = torch.arange(valid_matrix.nnz()).to(device)
                        else:
                            valid_combined_matrix, valid_mask = combine_matrices(valid_matrix, pred_idx_matrix)
                        valid_masks[target_rel.id] = valid_mask
                        data_target[target_rel.id] = valid_combined_matrix
                    else:
                        data_target[target_rel.id] = valid_matrix

                if use_equiv:
                    data_target.zero_()
                    idx_id_val, idx_trans_val = data_target.calculate_indices()
                    output_data = net(data, indices_identity, indices_transpose,
                               data_embedding, data_target, idx_id_val, idx_trans_val)
                else:
                    output_data = net(data, indices_identity, indices_transpose,
                                 data_embedding, data_target)
                logits_combined = torch.Tensor([]).to(device)
                for target_rel in target_rels:
                    logits_rel_full = output_data[target_rel.id].values.squeeze()
                    if use_equiv:
                        logits_rel = logits_rel_full[valid_masks[target_rel.id]]
                    else:
                        logits_rel = logits_rel_full
                    logits_combined = torch.cat([logits_combined, logits_rel])

                logp = torch.sigmoid(logits_combined)
                val_loss = loss_func(logp, labels_val).item()

                wandb_log.update({'val_loss': val_loss})
                left = left.cpu().numpy()
                right = right.cpu().numpy()
                edge_list = np.concatenate([left.reshape((1,-1)), right.reshape((1,-1))], axis=0)
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
        for target_rel in target_rels:
            print("Evaluating Target Rel " + str(target_rel.id))
            checkpoint = torch.load(checkpoint_path, map_location=device)
            net.load_state_dict(checkpoint['net_state_dict'])
            net.eval()

            # Target is same as input
            data_target = data.clone()
            with torch.no_grad():
                left_full, right_full, test_labels_full = get_test_neigh_from_file(dl, args.dataset, target_rel.id)
                test_matrix_full =  make_target_matrix_test(target_rel, left_full, right_full,
                                                      test_labels_full, device)
                test_matrix, left, right, test_labels = coalesce_matrix(test_matrix_full)
                if use_equiv:
                    test_combined_matrix, test_mask = combine_matrices(test_matrix, train_matrix)
                    data_target[target_rel.id] = test_combined_matrix
                    data_target.zero_()
                    idx_id_tst, idx_trans_tst = data_target.calculate_indices()
                    data_out = net(data, indices_identity, indices_transpose,
                               data_embedding, data_target, idx_id_tst, idx_trans_tst)
                    logits_full = data_out[target_rel.id].values.squeeze()
                    logits = logits_full[test_mask]
                else:
                    data_target[target_rel.id] = test_matrix
                    data_out = net(data, indices_identity, indices_transpose,
                                 data_embedding, data_target)
                    logits_full = data_out[target_rel.id].values.squeeze()
                    logits = logits_full

                pred = torch.sigmoid(logits).cpu().numpy()
                left = left.cpu().numpy()
                right = right.cpu().numpy()
                edge_list = np.vstack((left,right))
                edge_list_full = np.vstack((left_full, right_full))
                file_path = f"test_out/{run_name}.txt"
                gen_file_for_evaluate(dl, edge_list_full, edge_list, pred, target_rel.id,
                                         file_path=file_path)

#%% 
if __name__ == '__main__':
    argv = sys.argv[1:]
    args = get_hyperparams(argv)
    set_seed(args.seed)
    #%%
    run_model(args)
