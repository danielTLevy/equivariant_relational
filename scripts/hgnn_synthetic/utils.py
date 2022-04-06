# -*- coding: utf-8 -*-

import numpy as np
import random
import torch
import argparse
from sklearn.metrics import f1_score
from src.SparseMatrix import SparseMatrix
from sklearn.metrics import roc_auc_score
from collections import defaultdict

def set_seed(seed):
    random.seed(seed, version=2)
    np.random.seed(random.randint(0, 2**32))
    torch.manual_seed(random.randint(0, 2**32))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_hyperparams_nc(argv):
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
    ap.add_argument('--norm_out', type=int, default=0)
    ap.add_argument('--pool_op', type=str, default='mean')
    ap.add_argument('--use_edge_data',  type=int, default=1)
    ap.add_argument('--use_node_attr',  type=int, default=1)
    ap.add_argument('--residual', type=int, default=0)
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
    ap.add_argument('--sharing', action='store_true', default=False)

    ap.add_argument('--n_ents', type=int, default=1)
    ap.add_argument('--n_rels', type=int, default=1)
    ap.add_argument('--data_embed', type=int, default=10)
    ap.add_argument('--n_instances', type=int, default=1000)
    ap.add_argument('--sparsity', type=float, default=0.01)
    ap.add_argument('--p_het', type=float, default=0)
    ap.add_argument('--n_classes', type=int, default=3)
    ap.add_argument('--pct_val', type=float, default=0.2)
    ap.add_argument('--pct_test', type=float, default=0.2)
    ap.add_argument('--gen_links', type=str, default='uniform')
    ap.add_argument('--node_label', type=str, default='weight')
    ap.add_argument('--schema_str', type=str, default='')
    ap.add_argument('--node_attr', type=int, default=0)
    ap.set_defaults(wandb_log_run=False)

    args, argv = ap.parse_known_args(argv)
    if args.residual == 1:
        args.residual = True
    else:
        args.residual = False
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

def get_hyperparams_lp(argv):
    ap = argparse.ArgumentParser(allow_abbrev=False, description='EquivHGN for Link Prediction')
    ap.add_argument('--feats_type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others);' + 
                        '5 - only term features (zero vec for others).')
    ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
    ap.add_argument('--batch_size', type=int, default=100000)
    ap.add_argument('--patience', type=int, default=30, help='Patience.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training \
                    and testing for N times. Default is 1.')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--checkpoint_path', type=str, default='')
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--embedding_dim', type=int, default=64)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--act_fn', type=str, default='LeakyReLU')
    ap.add_argument('--in_fc_layer', type=int, default=1)
    ap.add_argument('--out_fc_layer', type=int, default=1)
    ap.add_argument('--optimizer', type=str, default='Adam')
    ap.add_argument('--val_every', type=int, default=5)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--norm_affine', type=int, default=1)
    ap.add_argument('--norm_embed', action='store_true', default=False)
    ap.add_argument('--pool_op', type=str, default='mean')
    ap.add_argument('--use_edge_data',  type=int, default=0)
    ap.add_argument('--use_other_edges',  type=int, default=1,
                    help='Whether to use non-target relations at all')
    ap.add_argument('--use_node_attrs',  type=int, default=1)
    ap.add_argument('--tail_weighted', type=int, default=0,
                    help='Whether to weight negative tail samples by frequency')
    ap.add_argument('--node_val',  type=str, default='one', help='Default value to use if nodes have no attributes')
    ap.add_argument('--pred_indices', type=str, default='train', help='Additional indices to include when making predictions')
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
    ap.add_argument('--lgnn', action='store_true', default=False)
    ap.add_argument('--alternating', action='store_true', default=False)
    ap.add_argument('--residual', type=int, default=0)
    ap.add_argument("--removed_params", type=int, nargs='*', default=None)
    ap.add_argument('--sharing', action='store_true', default=False)

    ap.add_argument('--n_ents', type=int, default=1)
    ap.add_argument('--n_rels', type=int, default=1)
    ap.add_argument('--data_embed', type=int, default=10)
    ap.add_argument('--n_instances', type=int, default=1000)
    ap.add_argument('--sparsity', type=float, default=0.01)
    ap.add_argument('--p_het', type=float, default=0)
    ap.add_argument('--pct_val', type=float, default=0.2)
    ap.add_argument('--pct_test', type=float, default=0.2)
    ap.add_argument('--gen_links', type=str, default='uniform')
    ap.add_argument('--schema_str', type=str, default='')
    ap.add_argument('--node_attr', type=int, default=0)
    ap.set_defaults(wandb_log_run=False)
    args, argv = ap.parse_known_args(argv)
    if args.residual == 1:
        args.residual = True
    else:
        args.residual = False
    if args.in_fc_layer == 1:
        args.in_fc_layer = True
    else:
        args.in_fc_layer = False
    if args.out_fc_layer == 1:
        args.out_fc_layer = True
    else:
        args.out_fc_layer = False
    if args.evaluate == 1:
        args.evaluate = True
    else:
        args.evaluate = False
    if args.norm_affine == 1:
        args.norm_affine = True
    else:
        args.norm_affine = False
    if args.norm_out == 1:
        args.norm_out = True
    else:
        args.norm_out = False
    if args.use_edge_data == 1:
        args.use_edge_data = True
    else:
        args.use_edge_data = False
    if args.use_other_edges == 1:
        args.use_other_edges = True
    else:
        args.use_edge_data = False
    if args.use_node_attrs == 1:
        args.use_node_attrs = True
    else:
        args.use_node_attrs = False
    if args.tail_weighted == 1:
        args.tail_weighted = True
    else:
        args.tail_weighted = False
    args.layers = [args.width]*args.depth
    return args

def select_features(data, schema, feats_type, target_ent=0):
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

def make_target_matrix_test(relation, left, right, labels, device):
    indices = torch.LongTensor(np.vstack((left, right)))
    values = torch.FloatTensor(labels).unsqueeze(1)
    shape = (relation.entities[0].n_instances,
             relation.entities[1].n_instances, 1)
    return SparseMatrix(indices=indices, values=values, shape=shape).to(device)

def evaluate_lp(edge_list, confidence, labels):
    """
    :param edge_list: shape(2, edge_num)
    :param confidence: shape(edge_num,)
    :param labels: shape(edge_num,)
    :return: dict with all scores we need
    """
    confidence = np.array(confidence)
    labels = np.array(labels)
    roc_auc = roc_auc_score(labels, confidence)
    mrr_list, cur_mrr = [], 0
    t_dict, labels_dict, conf_dict = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, h_id in enumerate(edge_list[0]):
        t_dict[h_id].append(edge_list[1][i])
        labels_dict[h_id].append(labels[i])
        conf_dict[h_id].append(confidence[i])
    for h_id in t_dict.keys():
        conf_array = np.array(conf_dict[h_id])
        rank = np.argsort(-conf_array)
        sorted_label_array = np.array(labels_dict[h_id])[rank]
        pos_index = np.where(sorted_label_array == 1)[0]
        if len(pos_index) == 0:
            continue
        pos_min_rank = np.min(pos_index)
        cur_mrr = 1 / (1 + pos_min_rank)
        mrr_list.append(cur_mrr)
    mrr = np.mean(mrr_list)

    return {'roc_auc': roc_auc, 'MRR': mrr}
