# -*- coding: utf-8 -*-


import sys
sys.path.append('../../')
sys.path.append('../')
from data.synthetic_heterogeneous import SyntheticHG

import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse
import dgl
from dgl.nn.pytorch import GraphConv, GATConv


from utils import get_hyperparams_nc, set_seed, select_features, f1_scores
from src.utils import count_parameters
import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterDict is not supported.")

class GAT(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, features_list):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(num_hidden, num_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features_list):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        return h
    

#%%
def run_model(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dl = SyntheticHG(args.n_ents, args.n_rels, args.data_embed,
                     args.n_instances, args.sparsity, args.p_het,
                     gen_links=args.gen_links,
                     schema_str=args.schema_str,
                     node_attr=args.node_attr)
    data, features_list = dl.to_edges_and_vals()
    print(dl.schema)
    print("Heterogeneous: {}".format(dl.signatures))
    dl.make_node_classification_task(args.n_classes, args.pct_test,
                                     args.pct_val, args.node_label)
    data = dl.data
    target_entity_id = 0
    target_entity = dl.schema.entities[target_entity_id]

    #data, in_dims = select_features(dl.data, dl.schema, args.feats_type, target_entity_id)
    features_list  = [feature.to(device) for feature in features_list]
    in_dims  = [features.shape[1] for features in features_list]
    labels = torch.LongTensor(dl.labels).to(device)
    train_idx = np.sort(dl.train_idx)
    val_idx = np.sort(dl.val_idx)
    test_idx = np.sort(dl.test_idx)

    adjM = scipy.sparse.csr_matrix(data[0].to_scipy_sparse())
    g = dgl.DGLGraph(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)

    feats_type = args.feats_type
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.gcn:
        net = GCN(g, in_dims=in_dims, num_hidden=args.width,
                  num_classes=args.n_classes, num_layers=args.depth, 
                  activation=eval('nn.%s()' % args.act_fn),dropout=args.dropout)
    elif args.gat:
        heads = [args.num_heads] * args.depth + [1]
        net = GAT(g, in_dims=in_dims, num_hidden=args.width,
                  num_classes=args.n_classes, num_layers=args.depth, heads=heads,
                  activation=eval('nn.%s()' % args.act_fn),
                  feat_drop=args.dropout, attn_drop=args.dropout,
                  negative_slope=args.slope, residual=False)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)


    if args.wandb_log_run:
        wandb.init(config=args,
            settings=wandb.Settings(start_method='fork'),
            project="EquivariantHGN_Synth_NC",
            entity='danieltlevy')
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
    progress = tqdm(range(args.epoch), desc="Epoch 0", position=0, leave=True)
    # training loop
    net.train()
    val_micro_best = 0
    for epoch in progress:
        
        # TODO: GET FEATURES LIST
        # training
        net.train()
        optimizer.zero_grad()
        logits = net(features_list)
        logp = F.log_softmax(logits, 1)
        train_loss = F.nll_loss(logp[train_idx], labels[train_idx])
        train_loss.backward()
        optimizer.step()
        train_micro, train_macro = f1_scores(logits[train_idx],
                                                 labels[train_idx])
        with torch.no_grad():
            progress.set_description(f"Epoch {epoch}")
            progress.set_postfix(loss=train_loss.item(), micr=train_micro)
            wandb_log = {'Train Loss': train_loss.item(),
                         'Train Micro': train_micro,
                         'Train Macro': train_macro}
            if epoch % args.val_every == 0:
                # validation
                net.eval()
                logits = net(features_list)
                logp = F.log_softmax(logits, 1)
                val_loss = F.nll_loss(logp[val_idx], labels[val_idx])
                val_micro, val_macro = f1_scores(logits[val_idx],
                                                         labels[val_idx])
                print("\nVal Loss: {:.3f} Val Micro-F1: {:.3f} \
Val Macro-F1: {:.3f}".format(val_loss, val_micro, val_macro))
                wandb_log.update({'Val Loss': val_loss.item(),
                                  'Val Micro-F1': val_micro, 'Val Macro-F1': val_macro})
                if val_micro > val_micro_best:

                    val_micro_best = val_micro
                    print("New best, saving")
                    torch.save({
                        'epoch': epoch,
                        'net_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss.item(),
                        'train_micro': train_micro,
                        'train_macro': train_macro,
                        'val_loss': val_loss.item(),
                        'val_micro': val_micro,
                        'val_macro': val_macro
                        }, checkpoint_path)
                    if args.wandb_log_run:
                        wandb.summary["val_micro_best"] = val_micro
                        wandb.summary["val_macro_best"] = val_macro
                        wandb.summary["val_loss_best"] = val_loss.item()
                        wandb.summary["epoch_best"] = epoch
                        wandb.summary["train_loss_best"] = train_loss.item()
                        wandb.summary['train_micro_best'] = train_micro
                        wandb.summary['train_macro_best'] = train_macro
                        wandb.save(checkpoint_path)

            if epoch % args.wandb_log_loss_freq == 0:
                if args.wandb_log_run:
                    wandb.log(wandb_log, step=epoch)


    # testing on test set
    if args.evaluate:
        pass
    wandb.finish()

#%%
if __name__ == '__main__':
    argv = sys.argv[1:]
    args = get_hyperparams_nc(argv)
    set_seed(args.seed)
    args.lgnn = True
    if args.model == 'gat':
        args.gat = True
        args.gcn = False
    else:
        args.gat = False
        args.gcn = True
    #%%
    run_model(args)
