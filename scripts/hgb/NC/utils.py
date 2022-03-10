# -*- coding: utf-8 -*-

import numpy as np
import torch
import argparse

def get_hyperparams(argv):
    ap = argparse.ArgumentParser(allow_abbrev=False,
                                 description='EquivHGN for Node Classification')
    ap.set_defaults(dataset='PubMed')
    ap.add_argument('--feats_type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others);' + 
                        '5 - only term features (zero vec for others).')
    ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=30, help='Patience.')
    ap.add_argument('--repeat', type=int, default=1,
                    help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--dataset', type=str, default='IMDB')
    ap.add_argument('--checkpoint_path', type=str, default='')
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--fc_layer', type=int, default=64)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--act_fn', type=str, default='LeakyReLU')
    ap.add_argument('--in_fc_layer', type=int, default=1)
    ap.add_argument('--mid_fc_layer', type=int, default=0)
    ap.add_argument('--optimizer', type=str, default='Adam')
    ap.add_argument('--val_every', type=int, default=5)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--norm',  type=int, default=1)
    ap.add_argument('--norm_affine', type=int, default=1)
    ap.add_argument('--norm_out', action='store_true', default=False)
    ap.add_argument('--pool_op', type=str, default='mean')
    ap.add_argument('--use_edge_data',  type=int, default=1)
    ap.add_argument('--use_node_attr',  type=int, default=1)
    ap.add_argument('--residual', action='store_true', default=False)
    ap.add_argument('--save_embeddings', dest='save_embeddings',
                    action='store_true', default=True)
    ap.add_argument('--no_save_embeddings', dest='save_embeddings',
                    action='store_false', default=True)
    ap.set_defaults(save_embeddings=True)
    ap.add_argument('--wandb_log_param_freq', type=int, default=1000)
    ap.add_argument('--wandb_log_loss_freq', type=int, default=1)
    ap.add_argument('--wandb_log_run', dest='wandb_log_run', action='store_true',
                        help='Log this run in wandb')
    ap.add_argument('--wandb_no_log_run', dest='wandb_log_run', action='store_false',
                        help='Do not log this run in wandb')
    ap.add_argument('--output', type=str)
    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--multi_label', default=False, action='store_true',
                    help='multi-label classification. Only valid for IMDb dataset')
    ap.add_argument('--evaluate', type=int, default=1)
    ap.add_argument('--lgnn', action='store_true', default=False)
    ap.add_argument("--removed_params", type=int, nargs='*', default=None)
    ap.add_argument("--asymmetric", action='store_true', default=False)
    ap.add_argument('--alternating', action='store_true', default=False)
    ap.add_argument('--embedding_dim', type=int, default=32)
    ap.set_defaults(wandb_log_run=False)

    args, argv = ap.parse_known_args(argv)

    if args.output == None:
        args.output = args.dataset + '_emb.dat'
    if args.dataset == 'IMDB':
        args.multi_label = True
    if args.in_fc_layer == 1:
        args.in_fc_layer = True
    else:
        args.in_fc_layer = False
    if args.mid_fc_layer == 1:
        args.mid_fc_layer = True
    else:
        args.mid_fc_layer = False
    if args.evaluate == 1:
        args.evaluate = True
    else:
        args.evaluate = False
    if args.norm_affine == 1:
        args.norm_affine = True
    else:
        args.norm_affine = False
    if args.norm == 1:
        args.norm = True
    else:
        args.norm = False
    if args.use_edge_data == 1:
        args.use_edge_data = True
    else:
        args.use_edge_data = False
    if args.use_node_attr == 1:
        args.use_node_attr = True
    else:
        args.use_node_attr = False
    args.layers = [args.width]*args.depth
    if args.fc_layer == 0:
        args.fc_layers = []
    else:
        args.fc_layers = [args.fc_layer]

    return args



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False, delta=0, save_path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss
