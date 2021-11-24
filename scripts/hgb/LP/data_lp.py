import networkx as nx
import numpy as np
import scipy
import pickle
import torch
import scipy.sparse as sp
import os
from collections import defaultdict
from data_loader_lp import data_loader
from src.DataSchema import DataSchema, Entity, Relation, SparseMatrixData
from src.SparseMatrix import SparseMatrix


DATA_FILE_DIR = '../../../data/hgb/LP/'

def load_data(prefix, use_node_attrs=True, use_edge_data=True, use_other_edges=True, node_val='one'):
    dl = data_loader(DATA_FILE_DIR+prefix)


    all_entities = [Entity(entity_id, n_instances)
                    for entity_id, n_instances
                    in sorted(dl.nodes['count'].items())]

    relations = {}
    test_types = dl.test_types
    if use_other_edges:
        for rel_id, (entity_i, entity_j) in sorted(dl.links['meta'].items()):
            relations[rel_id] = Relation(rel_id, [all_entities[entity_i], all_entities[entity_j]])

    else:
        for rel_id in test_types:
            entity_i, entity_j = dl.links['meta'][rel_id]
            relations[rel_id]  = Relation(rel_id, [all_entities[entity_i], all_entities[entity_j]])

    if use_other_edges:
        entities = all_entities
    else:
        entities = list(np.unique(relations[test_types[0]].entities))

    max_relation = max(relations) + 1
    if use_node_attrs:
        # Create fake relations to represent node attributes
        for entity in entities:
            rel_id = max_relation + entity.id
            relations[rel_id] = Relation(rel_id, [entity, entity], is_set=True)
    schema = DataSchema(entities, relations)

    data = SparseMatrixData(schema)
    for rel_id, data_matrix in dl.links['data'].items():
        if use_other_edges or rel_id in test_types:
            # Get subset belonging to entities in relation
            relation = relations[rel_id]
            start_i = dl.nodes['shift'][relation.entities[0].id]
            end_i = start_i + dl.nodes['count'][relation.entities[0].id]
            start_j = dl.nodes['shift'][relation.entities[1].id]
            end_j = start_j + dl.nodes['count'][relation.entities[1].id]
            rel_matrix = data_matrix[start_i:end_i, start_j:end_j]
            data[rel_id] = SparseMatrix.from_scipy_sparse(rel_matrix.tocoo())
            if not use_edge_data:
                # Use only adjacency information
                data[rel_id].values = torch.ones(data[rel_id].values.shape)

    if use_node_attrs:
        for ent in entities:
            ent_id = ent.id
            attr_matrix = dl.nodes['attr'][ent_id]
            n_instances = dl.nodes['count'][ent_id]
            if attr_matrix is None:
                if node_val == 'zero':
                    attr_matrix = np.zeros((n_instances,1))
                elif node_val == 'rand':
                    attr_matrix = np.random.randn(n_instances, 1)
                else:
                    attr_matrix = np.ones((n_instances,1))
            n_channels = attr_matrix.shape[1]
            rel_id = ent_id + max_relation
            indices = torch.arange(n_instances).unsqueeze(0).repeat(2, 1)
            data[rel_id] = SparseMatrix(
                indices = indices,
                values = torch.FloatTensor(attr_matrix),
                shape = np.array([n_instances, n_instances, n_channels]),
                is_set = True)

    return schema,\
           data, \
           dl

def load_data_flat(prefix, use_node_attrs=True, use_edge_data=True, node_val='one'):
    '''
    Load data into one matrix with all relations, reproducing Maron 2019
    The first [# relation types] channels are adjacency matrices,
    while the next [sum of feature dimensions per entity type] channels have
    node attributes on the relevant segment of their diagonals if use_node_attrs=True.
    If node features aren't included, then ndoe_val is used instead.
    '''
    dl = data_loader(DATA_FILE_DIR+prefix)
    total_n_nodes = dl.nodes['total']
    entities = [Entity(0, total_n_nodes)]
    relations = {0: Relation(0, [entities[0], entities[0]])}
    schema = DataSchema(entities, relations)

    # Sparse Matrix containing all data
    data_full =  sum(dl.links['data'].values()).tocoo()
    data_diag = scipy.sparse.coo_matrix((np.ones(total_n_nodes),
                             (np.arange(total_n_nodes),
                              np.arange(total_n_nodes))),
                            (total_n_nodes,total_n_nodes))
    data_full += data_diag
    data_full = SparseMatrix.from_scipy_sparse(data_full.tocoo()).zero_()
    data_out = SparseMatrix.from_other_sparse_matrix(data_full, 0)
    # Load up all edge data
    for rel_id in sorted(dl.links['data'].keys()):
        data_matrix = dl.links['data'][rel_id]
        data_rel =  SparseMatrix.from_scipy_sparse(data_matrix.tocoo())
        if not use_edge_data:
            # Use only adjacency information
            data_rel.values = torch.ones(data_rel.values.shape)
        data_rel_full = SparseMatrix.from_other_sparse_matrix(data_full, 1) + data_rel
        data_out.values = torch.cat([data_out.values, data_rel_full.values], 1)
        data_out.n_channels += 1

    if use_node_attrs:
        for ent_id, attr_matrix in dl.nodes['attr'].items():
            start_i = dl.nodes['shift'][ent_id]
            n_instances = dl.nodes['count'][ent_id]
            if attr_matrix is None:
                if node_val == 'zero':
                    attr_matrix = np.zeros((n_instances,1))
                elif node_val == 'rand':
                    attr_matrix = np.random.randn(n_instances, 1)
                else: 
                    attr_matrix = np.ones((n_instances,1))
            n_channels = attr_matrix.shape[1]
            indices = torch.arange(start_i, start_i+n_instances).unsqueeze(0).repeat(2, 1)
            data_rel = SparseMatrix(
                indices = indices,
                values = torch.FloatTensor(attr_matrix),
                shape = np.array([total_n_nodes, total_n_nodes, n_channels]),
                is_set = True)
            data_rel_full = SparseMatrix.from_other_sparse_matrix(data_full, n_channels) + data_rel
            data_out.values = torch.cat([data_out.values, data_rel_full.values], 1)
            data_out.n_channels += n_channels

    data = SparseMatrixData(schema)
    data[0] = data_out

    return schema,\
           data, \
           dl

def get_shifts(dl, edge_type, flat=False):
    '''
    Get shift needed to adjust indices of each node type
    If flat=True, simply return a shift of 0
    '''
    if flat:
        return 0, 0
    ent_i = dl.links['meta'][edge_type][0]
    ent_j = dl.links['meta'][edge_type][1]
    shift_i = dl.nodes['shift'][ent_i]
    shift_j = dl.nodes['shift'][ent_j]   
    return shift_i, shift_j

def get_train_valid_pos(dl, edge_type, flat=False):
    train_pos, valid_pos = dl.get_train_valid_pos()
    train_pos_arr = np.array(train_pos[edge_type])
    valid_pos_arr = np.array(valid_pos[edge_type])
    shift_i, shift_j = get_shifts(dl, edge_type, flat)
    train_pos_head_full = train_pos_arr[0] - shift_i
    train_pos_tail_full = train_pos_arr[1] - shift_j
    valid_pos_head_full = valid_pos_arr[0] - shift_i
    valid_pos_tail_full = valid_pos_arr[1] - shift_j
    return train_pos_head_full, train_pos_tail_full, \
            valid_pos_head_full, valid_pos_tail_full

def get_train_neg(dl, edge_type=None, edge_types=None, flat=False, tail_weighted=False):
    if edge_types is None:
        if edge_type is None:
            edge_types = []
        else:
            edge_types = [edge_type]
    train_neg_arr = np.array(dl.get_train_neg(edge_types,tail_weighted=tail_weighted)[edge_type])
    shift_i, shift_j = get_shifts(dl, edge_type, flat)
    train_neg_head = train_neg_arr[0] - shift_i
    train_neg_tail = train_neg_arr[1] - shift_j
    return train_neg_head, train_neg_tail


def get_valid_neg(dl, edge_type=None, edge_types=None, flat=False):
    if edge_types is None:
        if edge_type is None:
            edge_types = []
        else:
            edge_types = [edge_type]
    train_neg_arr = np.array(dl.get_valid_neg(edge_types)[edge_type])
    shift_i, shift_j = get_shifts(dl, edge_type, flat)
    valid_neg_head = train_neg_arr[0] - shift_i
    valid_neg_tail = train_neg_arr[1] - shift_j
    return valid_neg_head, valid_neg_tail

def get_valid_neg_2hop(dl, edge_type, flat=False):
    train_neg_arr = np.array(dl.get_valid_neg_2hop(edge_type))
    shift_i, shift_j = get_shifts(dl, edge_type, flat)
    valid_neg_head = train_neg_arr[0] - shift_i
    valid_neg_tail = train_neg_arr[1] - shift_j
    return valid_neg_head, valid_neg_tail

def get_test_neigh(dl, edge_type=None, neigh_type=None, flat=False):
    if neigh_type == 'w_random':
        get_test_neigh = dl.get_test_neigh_w_random
    elif neigh_type == 'full_random':
        get_test_neigh = dl.get_test_neigh_full_random
    else:
        get_test_neigh = dl.get_test_neigh
    test_neigh, test_label = get_test_neigh()
    test_neigh_arr = np.array(test_neigh[edge_type])
    test_label_arr = np.array(test_label[edge_type])
    shift_i, shift_j = get_shifts(dl, edge_type, flat)
    test_neigh_head = test_neigh_arr[0] - shift_i
    test_neigh_tail = test_neigh_arr[1] - shift_j
    return test_neigh_head, test_neigh_tail, test_label_arr

def get_test_neigh_from_file(dl, dataset, edge_type, flat=False):
    save = np.loadtxt(os.path.join(dl.path, f"{dataset}_ini_{edge_type}_label.txt"), dtype=int)
    shift_i, shift_j = get_shifts(dl, edge_type, flat)

    left  = save[0] - shift_i
    right = save[1] - shift_j
    # Don't have access to real labels, just get random
    test_label = np.random.randint(2, size=save[0].shape[0])
    return left, right, test_label



def gen_file_for_evaluate(dl, target_edges, edges, confidence, edge_type, file_path, flat=False):
    """
    :param edge_list: shape(2, edge_num)
    :param confidence: shape(edge_num,)
    :param edge_type: shape(1)
    :param file_path: string
    """
    # First, turn output into dict
    output_dict = defaultdict(dict)
    shift_l, shift_r = get_shifts(dl, edge_type, flat)
    for l,r,c in zip(edges[0], edges[1], confidence):
            l_i = l + shift_l
            r_i = r + shift_r
            output_dict[l_i][r_i] = c
    # Then, write all the target test edges
    with open(file_path, "a") as f:
        for l,r in zip(target_edges[0], target_edges[1]):
            l_i = l + shift_l
            r_i = r + shift_r
            c = output_dict[l_i][r_i]
            f.write(f"{l_i}\t{r_i}\t{edge_type}\t{c}\n")
