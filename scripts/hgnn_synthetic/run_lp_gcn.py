import sys
sys.path.append('../../')
sys.path.append('../')

import torch
import torch.nn as nn
import numpy as np
import wandb
from tqdm import tqdm
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
import scipy.sparse
import dgl
import pdb


from data.synthetic_heterogeneous import SyntheticHG
from hgb.LP.EquivHGAE import EquivLinkPredictor, EquivLinkPredictorShared
from src.DataSchema import SparseMatrixData
from src.utils import count_parameters
from utils import get_hyperparams_lp, set_seed, select_features, \
    make_target_matrix, coalesce_matrix, combine_matrices, evaluate_lp
import warnings

warnings.filterwarnings("ignore", message="Setting attributes on ParameterDict is not supported.")

#%%
class DisMult(nn.Module):
    def __init__(self, rel_num, dim):
        super(DisMult, self).__init__()
        self.dim = dim
        self.weights = nn.Parameter(torch.FloatTensor(size=(dim, dim)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weights, gain=1.414)

    def forward(self, input1, input2):
        input1 = torch.unsqueeze(input1, 1)
        input2 = torch.unsqueeze(input2, 2)
        tmp = torch.bmm(input1, self.weights)
        re = torch.bmm(tmp, input2).squeeze()
        return re


class Dot(nn.Module):
    def __init__(self):
        super(Dot, self).__init__()

    def forward(self, input1, input2):
        input1 = torch.unsqueeze(input1, 1)
        input2 = torch.unsqueeze(input2, 2)
        return torch.bmm(input1, input2).squeeze()


class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats, activation, n_layers=2, dropout=0.5, decoder='dot', rel_num=1):
        super(GCN, self).__init__()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hid_feats, bias=True) for feats_dim in in_feats])
        #for fc in self.fc_list:
        #    nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNConv(hid_feats, hid_feats))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNConv(hid_feats, hid_feats))
        self.dropout = nn.Dropout(p=dropout)
        #for layer in self.layers:
        #    nn.init.xavier_normal_(layer.lin.weight, gain=1.4)
        if decoder == 'dismult':
            self.decode = DisMult(rel_num=rel_num, dim=hid_feats)
        elif decoder == 'dot':
            self.decode = Dot()

    def encode(self, data):
        x, edge_list = data
        h = []
        for fc, feature in zip(self.fc_list, x):
            h.append(fc(feature))
        x = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
            x = layer(x, edge_list)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x


class GAT(nn.Module):
    def __init__(self, in_feats, hid_feats, activation, n_layers=2, dropout=0.5, heads=[1], decoder="dot", rel_num=1):
        super(GAT, self).__init__()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hid_feats, bias=True) for feats_dim in in_feats])
        #for fc in self.fc_list:
        #    nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GATConv(hid_feats, hid_feats, heads[0]))
        # hidden layers
        for l in range(1, n_layers - 1):
            self.layers.append(GATConv(hid_feats * heads[l - 1], hid_feats, heads[l]))
        # output layer
        self.layers.append(GATConv(hid_feats * heads[-2], hid_feats, heads[-1]))
        self.dropout = nn.Dropout(p=dropout)
        #for layer in self.layers:
        #    nn.init.xavier_normal_(layer.lin_l.weight, gain=1.4)
        if decoder == 'dismult':
            self.decode = DisMult(rel_num=rel_num, dim=hid_feats)
        elif decoder == 'dot':
            self.decode = Dot()

    def encode(self, data):
        x, edge_list = data
        h = []
        for fc, feature in zip(self.fc_list, x):
            h.append(fc(feature))
        x = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
            x = layer(x, edge_list)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x


#%%
def run_model(args):
    print(args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Collect data and schema
    dl = SyntheticHG(args.n_ents, args.n_rels, args.data_embed,
                     args.n_instances, args.sparsity, args.p_het,
                     gen_links=args.gen_links,
                     schema_str=args.schema_str,
                     node_attr=args.node_attr)
    data, features_list = dl.to_edges_and_vals()
    feat_list = []
    in_dims = []
    for ent_id in range(args.n_ents):
        if args.feats_type == 0:
            # For each node, include either all ones, or data features
            if len(features_list) == 0:
                feat = torch.ones(args.n_instances, 1)
            else:
                feat = features_list[ent_id]
            feat_list.append(feat.to(device))
            in_dims.append(feat.shape[1])
        elif args.feats_type == 3:
            # For each node, include sparse one-hot vector as a feature
            indices = np.vstack((np.arange(args.n_instances), np.arange(args.n_instances)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(args.n_instances))
            feat = torch.sparse.FloatTensor(indices, values,
                                         torch.Size([args.n_instances,
                                                     args.n_instances])).to(device).coalesce()
            feat_list.append(feat.to(device))
            in_dims.append(feat.values.shape[1])

    print(dl.full_schema)
    print("Heterogeneous: {}".format(dl.signatures))
    dl.make_link_prediction_task(args.pct_test, args.pct_val, args.val_neg, args.tail_weighted)
    dl.to_flat()

    edge_list = torch.LongTensor(dl.data[0].indices).to(device)
    features_list  = [feature.to(device) for feature in features_list]
    data = data.to(device)
    
    # Get target relations and create data structure for embeddings
    target_rel_id = dl.target_rel_id
    target_rel = dl.schema.relations[target_rel_id]
    target_ents = dl.schema.entities
    flat_rel = dl.schema.relations[0]
    target_ents = dl.schema.entities

    # Get training and validation positive samples now
    train_pos, val_pos = dl.get_train_valid_pos()
    train_pos_head, train_pos_tail = train_pos[0], train_pos[1]
    val_pos_head, val_pos_tail = val_pos[0], val_pos[1]

    # Create network and optimizer
    if args.gcn:
        net = GCN(in_feats=in_dims, hid_feats=args.width,
                    activation=eval('nn.%s()' % args.act_fn), 
                    n_layers=args.depth, dropout=args.dropout,
                    decoder=args.decoder)

    elif args.gat:
        heads = [args.num_heads] * args.depth + [1]
        net = GAT(in_feats=in_dims, hid_feats=args.width, 
                    activation=eval('nn.%s()' % args.act_fn), 
                    n_layers=args.depth, heads=heads,
                    dropout=args.dropout, decoder=args.decoder)


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
        labels_train = torch.Tensor([]).to(device)

        train_neg_head, train_neg_tail = dl.get_train_neg(tail_weighted=args.tail_weighted)


        train_matrix = make_target_matrix(flat_rel,
                                               train_pos_head,
                                               train_pos_tail,
                                               train_neg_head,
                                               train_neg_tail,
                                               device)
        data_target[flat_rel.id] = train_matrix
        train_heads, train_tails = train_matrix.indices[0], train_matrix.indices[1]
        labels_train = train_matrix.values[:,target_rel_id]
        # Make prediction
        hid_feat = net.encode([feat_list, edge_list])
        logits = net.decode(hid_feat[train_heads], hid_feat[train_tails])
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
                valid_neg_head, valid_neg_tail = dl.get_valid_neg(args.val_neg)

                valid_matrix_full = make_target_matrix(target_rel,
                                                 val_pos_head, val_pos_tail,
                                                 valid_neg_head, valid_neg_tail,
                                                 device)
                valid_matrix, val_heads, val_tails, labels_val = coalesce_matrix(valid_matrix_full)

                data_target[target_rel.id] = valid_matrix

                # Make prediction
                hid_feat = net.encode([feat_list, edge_list])
                logits = net.decode(hid_feat[val_heads], hid_feat[val_tails])
                logp = torch.sigmoid(logits)
                val_loss = loss_func(logp, labels_val).item()



                # Calculate ROC AUC and MRR
                val_heads = val_heads.cpu().numpy()
                val_tails = val_tails.cpu().numpy()
                eval_edge_list = np.concatenate([val_heads.reshape((1,-1)), val_tails.reshape((1,-1))], axis=0)
                res = evaluate_lp(eval_edge_list, logp.cpu().numpy(), labels_val.cpu().numpy())
                
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
        '''
        print("Evaluating Target Rel " + str(rel_id))
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net_state_dict'])
        net.eval()

        # Target is same as input
        data_target = data.clone()
        with torch.no_grad():
            test_heads_full, test_tails_full, test_labels_full = get_test_neigh_from_file(dl, args.dataset, rel_id, flat=True)

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
        '''
        pass
    wandb.finish()
#%%
if __name__ == '__main__':
    argv = sys.argv[1:]
    args = get_hyperparams_lp(argv)
    args.lgnn = True
    set_seed(args.seed)
    if args.model == 'gat':
        args.gat = True
        args.gcn = False
    else:
        args.gat = False
        args.gcn = True
    #%%
    run_model(args)
