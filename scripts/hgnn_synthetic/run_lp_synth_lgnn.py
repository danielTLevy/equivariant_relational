import sys
sys.path.append('../../')
sys.path.append('../')

import torch
import torch.nn as nn
import numpy as np
import wandb
from tqdm import tqdm

from data.synthetic_heterogeneous import SyntheticHG

from hgb.LP.EquivHGAE import EquivLinkPredictor
from src.SparseMatrix import SparseMatrix
from src.DataSchema import SparseMatrixData
from src.utils import count_parameters
from utils import get_hyperparams_lp, set_seed, select_features, \
    make_target_matrix, coalesce_matrix, combine_matrices, evaluate_lp, \
        make_target_matrix_test
import warnings

warnings.filterwarnings("ignore", message="Setting attributes on ParameterDict is not supported.")

#%%
def run_model(args):
    print(args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_equiv = args.decoder == 'equiv'

    # Collect data and schema
    dl = SyntheticHG(args.n_ents, args.n_rels, args.data_embed,
                     args.n_instances, args.sparsity, args.p_het,
                     gen_links=args.gen_links,
                     schema_str=args.schema_str,
                     node_attr=args.node_attr,
                     scaling=args.scaling)
    print(dl.full_schema)
    print("Heterogeneous: {}".format(dl.rel_functions))
    dl.make_link_prediction_task(args.pct_test, args.pct_val, args.val_neg, args.tail_weighted)
    dl.to_flat()

    data, in_dims = select_features(dl.data, dl.schema, args.feats_type)
    data = data.to(device)
    # Precompute data indices
    indices_identity, indices_transpose = data.calculate_indices()
    # Get target relations and create data structure for embeddings
    target_rel_id = dl.target_rel_id
    target_rel = dl.schema.relations[target_rel_id]
    target_ents = dl.schema.entities
    flat_rel = dl.schema.relations[0]
    target_ents = dl.schema.entities

    # Create a matrix containing just the positive indices
    data_blank = SparseMatrix.from_other_sparse_matrix(data[flat_rel.id], 1)

    # Create object for node embeddings
    data_embedding = SparseMatrixData.make_entity_embeddings(target_ents,
                                                             args.embedding_dim)
    data_embedding.to(device)

    # Get training and validation positive samples now
    train_pos, val_pos = dl.get_train_valid_pos()
    train_pos_head, train_pos_tail = train_pos[0], train_pos[1]
    val_pos_head, val_pos_tail = val_pos[0], val_pos[1]

    # Get additional indices to be used when making predictions
    if args.pred_indices == 'train':
        # Get positive links for all relation types
        all_pos_head = data[0].indices[0].cpu().numpy()
        all_pos_tail = data[0].indices[1].cpu().numpy()
        # Get negative links for target relation type
        train_neg_head, train_neg_tail = dl.get_train_neg(args.tail_weighted)
        pred_idx_matrix = make_target_matrix(target_rel,
                                          all_pos_head, all_pos_tail,
                                          train_neg_head, train_neg_tail,
                                          device)
    elif args.pred_indices == 'train_neg':
        # Get negative samples twice
        train_neg_head1, train_neg_tail1 = dl.get_train_neg(args.tail_weighted)
        train_neg_head2, train_neg_tail2 = dl.get_train_neg(args.tail_weighted)
        pred_idx_matrix= make_target_matrix(target_rel,
                                          train_neg_head1, train_neg_tail1,
                                          train_neg_head2, train_neg_tail2,
                                          device)
    elif args.pred_indices == 'none':
        pred_idx_matrix = None

    # Create network and optimizer
    net = EquivLinkPredictor(dl.schema, in_dims,
                    layers=args.layers,
                    embedding_dim=args.embedding_dim,
                    embedding_entities=target_ents,
                    output_rels=None,
                    activation=eval('nn.%s()' % args.act_fn),
                    final_activation = nn.Identity(),
                    dropout=args.dropout,
                    pool_op=args.pool_op,
                    norm_affine=args.norm_affine,
                    norm_embed=args.norm_embed,
                    in_fc_layer=args.in_fc_layer,
                    decode = 'equiv',
                    out_dim = 1)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Set up logging and checkpointing
    if args.wandb_log_run:
        wandb.init(config=args,
            settings=wandb.Settings(start_method='fork'),
            project="EquivariantHGN_Synth_LP",
            entity='danieltlevy',
            tags=args.tags)
        wandb.watch(net, log='all', log_freq=args.wandb_log_param_freq)
    print(args)
    print("Number of parameters: {}".format(count_parameters(net)))
    run_name = str(args.run)
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

        train_neg_head, train_neg_tail = dl.get_train_neg(tail_weighted=args.tail_weighted)


        train_matrix = make_target_matrix(flat_rel,
                                               train_pos_head,
                                               train_pos_tail,
                                               train_neg_head,
                                               train_neg_tail,
                                               device)
        labels_train = train_matrix.values[:,target_rel_id]

        # Add in all data indicted to target_matrix
        target_matrix, train_mask = combine_matrices(train_matrix, data_blank)
        data_target[flat_rel.id] = target_matrix


        # Make prediction
        idx_id_tgt, idx_trans_tgt = data_target.calculate_indices()
        output_data = net(data, indices_identity, indices_transpose,
                   data_embedding, data_target, idx_id_tgt, idx_trans_tgt)

        logits = output_data[flat_rel.id].values[train_mask, target_rel_id]
        logp = torch.sigmoid(logits)
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
                # Make validation matrix
                val_neg_head, val_neg_tail = dl.get_valid_neg(args.val_neg)
                valid_matrix_full = make_target_matrix(target_rel,
                                                 val_pos_head, val_pos_tail,
                                                 val_neg_head, val_neg_tail,
                                                 device)
                valid_matrix, _, _, _ = coalesce_matrix(valid_matrix_full)
                if use_equiv:
                    # Add in additional prediction indices
                    if pred_idx_matrix is None:
                        valid_combined_matrix = valid_matrix
                        valid_mask = torch.arange(valid_matrix.nnz()).to(device)
                    else:
                        valid_combined_matrix, valid_mask = combine_matrices(valid_matrix, pred_idx_matrix)
                    data_target[target_rel.id] = valid_combined_matrix
                else:
                    data_target[target_rel.id] = valid_matrix
                # Get validation indices and labels
                left, right = valid_combined_matrix.indices[:, valid_mask]
                labels_val = valid_combined_matrix.values[valid_mask,:].squeeze()

                # Make prediction
                data_target.zero_()
                idx_id_val, idx_trans_val = data_target.calculate_indices()
                output_data = net(data, indices_identity, indices_transpose,
                           data_embedding, data_target, idx_id_val, idx_trans_val)
                logits_full = output_data[target_rel_id].values.squeeze()
                if use_equiv:
                    logits = logits_full[valid_mask]
                else:
                    logits = logits_full
                logp = torch.sigmoid(logits)
                # Get loss
                val_loss = loss_func(logp, labels_val).item()

                # Calculate ROC AUC and MRR
                left = left.cpu().numpy()
                right = right.cpu().numpy()
                edge_list = np.concatenate([left.reshape((1,-1)), right.reshape((1,-1))], axis=0)
                res = evaluate_lp(edge_list, logp.cpu().numpy(), labels_val.cpu().numpy())
                
                val_roc_auc = res['roc_auc']
                val_mrr = res['MRR']
                wandb_log.update({'val_loss': val_loss})
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
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net_state_dict'])
        net.eval()

        # Target is same as input
        data_target = data.clone()
        with torch.no_grad():
            # Get test indices and deduplicate any repeated indices
            test_heads, test_tails, test_labels =  dl.get_test_neigh()
            test_matrix_full = make_target_matrix_test(target_rel,
                                             test_heads, test_tails,
                                             test_labels, device)
            test_matrix, left, right, test_labels = coalesce_matrix(test_matrix_full)
            if use_equiv:
                # Add in additional prediction indices
                if pred_idx_matrix is None:
                    test_combined_matrix = test_matrix
                    test_mask = torch.arange(test_matrix.nnz()).to(device)
                else:
                    test_combined_matrix, test_mask = combine_matrices(test_matrix, pred_idx_matrix)
                data_target[target_rel.id] = test_combined_matrix
            else:
                data_target[target_rel.id] = test_matrix

            data_target.zero_()
            idx_id_tst, idx_trans_tst = data_target.calculate_indices()
            output_data = net(data, indices_identity, indices_transpose,
                       data_embedding, data_target, idx_id_tst, idx_trans_tst)


            logits = output_data[flat_rel.id].values[:, target_rel_id][test_mask]
            pred = torch.sigmoid(logits).cpu().numpy()

            left = left.cpu().numpy()
            right = right.cpu().numpy()
            edge_list = np.vstack((left,right))
            res = evaluate_lp(edge_list, pred, test_labels.cpu().numpy())
            test_roc_auc = res['roc_auc']
            test_mrr = res['MRR']
            wandb_log.update(res)
            print("\nTest ROC AUC: {:.3f} Test MRR: {:.3f}".format(
                test_roc_auc, test_mrr))

            if args.wandb_log_run:
                wandb.summary['test_roc_auc'] = test_roc_auc
                wandb.summary['test_mrr'] = test_mrr

        pass
    wandb.finish()
#%%
if __name__ == '__main__':
    argv = sys.argv[1:]
    args = get_hyperparams_lp(argv)
    args.lgnn = True
    args.model = 'lgnn'
    set_seed(args.seed)
    #%%
    run_model(args)
