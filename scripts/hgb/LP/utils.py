import argparse
import random
import numpy as np
import torch
from src.SparseMatrix import SparseMatrix

def set_seed(seed):
    random.seed(seed, version=2)
    np.random.seed(random.randint(0, 2**32))
    torch.manual_seed(random.randint(0, 2**32))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def coalesce_matrix(matrix):
    coalesced_matrix = matrix.coalesce(op='mean')
    left = coalesced_matrix.indices[0,:]
    right = coalesced_matrix.indices[1,:]
    labels = coalesced_matrix.values.squeeze()
    return coalesced_matrix, left, right, labels

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

def make_target_matrix_test(relation, left, right, labels, device):
    indices = torch.LongTensor(np.vstack((left, right)))
    values = torch.FloatTensor(labels).unsqueeze(1)
    shape = (relation.entities[0].n_instances,
             relation.entities[1].n_instances, 1)
    return SparseMatrix(indices=indices, values=values, shape=shape).to(device)


def select_features(data, schema, feats_type):
    '''
    TODO: IMPLEMENT THIS
    '''
    # Select features for nodes
    in_dims = {}

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
    for rel_id in schema.relations:
        in_dims[rel_id] = data[rel_id].n_channels
    return data, in_dims

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
    ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
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
    ap.add_argument('--residual', action='store_true', default=False)
    ap.add_argument("--removed_params", type=int, nargs='*', default=None)
    ap.set_defaults(wandb_log_run=False)
    args, argv = ap.parse_known_args(argv)
    if args.output == None:
        args.output = args.dataset + '_emb.dat'
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
