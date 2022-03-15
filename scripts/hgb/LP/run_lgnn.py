import sys
sys.path.append('../../')

import torch
import torch.nn as nn
import numpy as np
import wandb
from tqdm import tqdm
from data_lp import load_data_flat, get_train_valid_pos, get_train_neg, \
    get_valid_neg, get_valid_neg_2hop, get_test_neigh_from_file, gen_file_for_evaluate
from EquivHGAE import EquivLinkPredictor
from src.DataSchema import SparseMatrixData
from src.utils import count_parameters
from utils import get_hyperparams, set_seed, select_features, combine_matrices_flat, make_flat_target_matrix
import warnings

warnings.filterwarnings("ignore", message="Setting attributes on ParameterDict is not supported.")

#%%
def run_model(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Collect data and schema
    schema, data_original, dl = load_data_flat(args.dataset,
                                               use_edge_data=args.use_edge_data,
                                               use_node_attrs=args.use_node_attrs,
                                               node_val=args.node_val)
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
                    output_rels=None,
                    activation=eval('nn.%s()' % args.act_fn),
                    final_activation = nn.Identity(),
                    dropout=args.dropout,
                    pool_op=args.pool_op,
                    norm_affine=args.norm_affine,
                    norm_embed=args.norm_embed,
                    in_fc_layer=args.in_fc_layer,
                    decode = 'equiv',
                    out_dim = num_outputs)
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
        # Target is same as input
        data_target = data.clone()
        labels_train = torch.Tensor([]).to(device)

        train_neg_heads, train_neg_tails = dict(), dict()
        for target_rel_id in target_rel_ids:
            train_neg_heads[target_rel_id], train_neg_tails[target_rel_id] = get_train_neg(dl, target_rel_id, flat=True,
                                                                                           tail_weighted=args.tail_weighted)


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
                        val_neg_heads[target_rel_id], val_neg_tails[target_rel_id] = get_valid_neg_2hop(dl, target_rel_id, flat=True)
                    elif args.val_neg == 'randomtw':
                        val_neg_heads[target_rel_id], val_neg_tails[target_rel_id] = get_valid_neg(dl, target_rel_id, tail_weighted=True, flat=True)
                    else:
                        val_neg_heads[target_rel_id], val_neg_tails[target_rel_id] = get_valid_neg(dl, target_rel_id, flat=True),

                val_matrix_combined, val_masks = combine_matrices_flat(flat_rel, val_pos_heads,
                                                    val_pos_tails, val_neg_heads,
                                                    val_neg_tails, target_rel_ids, train_matrix,
                                                    device)
                data_target[flat_rel.id] = val_matrix_combined.clone()


                data_target.zero_()
                idx_id_val, idx_trans_val = data_target.calculate_indices()
                output_data = net(data, indices_identity, indices_transpose,
                           data_embedding, data_target, idx_id_val, idx_trans_val)

                left = torch.Tensor([]).to(device)
                right = torch.Tensor([]).to(device)
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
                test_heads_full[rel_id], test_tails_full[rel_id], test_labels_full = get_test_neigh_from_file(dl, args.dataset, rel_id, flat=True)

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
if __name__ == '__main__':
    argv = sys.argv[1:]
    args = get_hyperparams(argv)
    args.lgnn = True
    set_seed(args.seed)
    #%%
    run_model(args)
