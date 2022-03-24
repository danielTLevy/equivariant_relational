# -*- coding: utf-8 -*-

import numpy as np
import random
import torch
import argparse
from sklearn.metrics import f1_score
from src.SparseMatrix import SparseMatrix
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed, version=2)
    np.random.seed(random.randint(0, 2**32))
    torch.manual_seed(random.randint(0, 2**32))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_hyperparams(argv):
    ap = argparse.ArgumentParser(allow_abbrev=False,
                                 description='EquivHGN for Node Classification')
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
    ap.add_argument('--evaluate', type=int, default=1)
    ap.add_argument('--lgnn', action='store_true', default=False)
    ap.add_argument("--removed_params", type=int, nargs='*', default=None)
    ap.add_argument('--alternating', action='store_true', default=False)
    ap.add_argument('--embedding_dim', type=int, default=32)
    ap.add_argument('--sharing', action='store_true', default=False)

    ap.add_argument('--n_ents', type=int, default=1)
    ap.add_argument('--n_rels', type=int, default=1)
    ap.add_argument('--data_embed', type=int, default=10)
    ap.add_argument('--n_instances', type=int, default=1000)
    ap.add_argument('--sparsity', type=float, default=0.01)
    ap.add_argument('--p_het', type=float, default=0)
    ap.add_argument('--n_classes', type=int, default=3)
    ap.add_argument('--pct_val', type=float, default=0.2)
    ap.add_argument('--pct_test', type=int, default=0.2)
    ap.add_argument('--gen_links', type=str, default='uniform')
    ap.set_defaults(wandb_log_run=False)

    args, argv = ap.parse_known_args(argv)

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

def select_features(data, schema, feats_type, target_ent):
    '''
    TODO: IMPLEMENT THIS
    '''
    # Select features for nodes
    in_dims = {}
    # Starting index of entity-relation
    ent_start = max(schema.relations.keys()) - len(schema.entities) + 1

    if feats_type == 0:
        # Keep all node attributes
        pass
    elif feats_type == 1:
        # Set all non-target node attributes to zero
        for ent_i in schema.entities:
            if ent_i.id != target_ent:
                # 10 dimensions for some reason
                n_dim = 10
                rel_id = ent_start + ent_i.id
                data[rel_id] = SparseMatrix.from_other_sparse_matrix(data[rel_id], n_dim)

    '''
    elif feats_type == 2:
        # Set all non-target node attributes to one-hot vector
        for i in range(0, len(features_list)):
            if i != target_ent:
                dim = features_list[i].shape[0]
                indices = torch.arange(n_instances).unsqueeze(0).repeat(2, 1)
                values = torch.FloatTensor(np.ones(dim))
                features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = np.ones(dim)
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    '''
    for rel_id in schema.relations:
        in_dims[rel_id] = data[rel_id].n_channels
    return data, in_dims

def f1_scores(logits, target):
    values = logits.argmax(1).detach().cpu()
    micro = f1_score(target.cpu(), values, average='micro')
    macro = f1_score(target.cpu(), values, average='macro')
    return micro, macro
