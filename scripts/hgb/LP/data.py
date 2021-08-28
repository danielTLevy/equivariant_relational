import networkx as nx
import numpy as np
import scipy
import pickle
import torch
import scipy.sparse as sp
from data_loader import data_loader
from src.DataSchema import DataSchema, Entity, Relation, SparseMatrixData
from src.SparseMatrix import SparseMatrix


DATA_FILE_DIR = '../../../data/hgb/LP/'

def load_data(prefix='LastFM'):
    dl = data_loader(DATA_FILE_DIR+prefix)

    entities = [Entity(entity_id, n_instances)
                for entity_id, n_instances
                in sorted(dl.nodes['count'].items())]
    relations = [Relation(rel_id, [entities[entity_i], entities[entity_j]])
                    for rel_id, (entity_i, entity_j)
                    in sorted(dl.links['meta'].items())]
    num_relations = len(relations)
    use_node_attrs = True
    if use_node_attrs:
        # Create fake relations to represent node attributes
        for entity in entities:
            rel_id = num_relations + entity.id
            rel = Relation(rel_id, [entity, entity], is_set=True)
            relations.append(rel)
    schema = DataSchema(entities, relations)

    data = SparseMatrixData(schema)
    for rel_id, data_matrix in dl.links['data'].items():
        # Get subset belonging to entities in relation
        start_i = dl.nodes['shift'][relations[rel_id].entities[0].id]
        end_i = start_i + dl.nodes['count'][relations[rel_id].entities[0].id]
        start_j = dl.nodes['shift'][relations[rel_id].entities[1].id]
        end_j = start_j + dl.nodes['count'][relations[rel_id].entities[1].id]
        rel_matrix = data_matrix[start_i:end_i, start_j:end_j]
        data[rel_id] = SparseMatrix.from_scipy_sparse(rel_matrix.tocoo())

    for ent_id, attr_matrix in dl.nodes['attr'].items():
        if attr_matrix is None:
            # Attribute for each node is a single 1
            attr_matrix = np.ones(dl.nodes['count'][ent_id])[:, None]
        n_channels = attr_matrix.shape[1]
        rel_id = ent_id + num_relations
        n_instances = dl.nodes['count'][ent_id]
        indices = torch.arange(n_instances).unsqueeze(0).repeat(2, 1)
        data[rel_id] = SparseMatrix(
            indices = indices,
            values = torch.FloatTensor(attr_matrix),
            shape = np.array([n_instances, n_instances, n_channels]),
            is_set = True)


    target_rel_id = dl.test_types[0]
    ent_i, ent_j = relations[target_rel_id].entities

    #TODO: this next line might cause an issue
    schema_out = DataSchema([ent_i, ent_j], relations[target_rel_id])

    return schema,\
           schema_out, \
           data, \
           dl

def get_shifts(dl, edge_type):
    ent_i = dl.links['meta'][edge_type][0]
    ent_j = dl.links['meta'][edge_type][1]
    shift_i = dl.nodes['shift'][ent_i]
    shift_j = dl.nodes['shift'][ent_j]   
    return shift_i, shift_j

def get_train_valid_pos(dl, edge_type):
    train_pos, valid_pos = dl.get_train_valid_pos()
    train_pos_arr = np.array(train_pos[edge_type])
    valid_pos_arr = np.array(valid_pos[edge_type])
    shift_i, shift_j = get_shifts(dl, edge_type)
    train_pos_head_full = train_pos_arr[0] - shift_i
    train_pos_tail_full = train_pos_arr[1] - shift_j
    valid_pos_head_full = valid_pos_arr[0] - shift_i
    valid_pos_tail_full = valid_pos_arr[1] - shift_j
    return train_pos_head_full, train_pos_tail_full, \
            valid_pos_head_full, valid_pos_tail_full

def get_train_neg(dl, edge_type=None, edge_types=None):
    if edge_types is None:
        if edge_type is None:
            edge_types = []
        else:
            edge_types = [edge_type]
    train_neg_arr = np.array(dl.get_train_neg(edge_types)[edge_type])
    shift_i, shift_j = get_shifts(dl, edge_type)
    train_neg_head = train_neg_arr[0] - shift_i
    train_neg_tail = train_neg_arr[1] - shift_j
    return train_neg_head, train_neg_tail

def get_valid_neg(dl, edge_type=None, edge_types=None):
    if edge_types is None:
        if edge_type is None:
            edge_types = []
        else:
            edge_types = [edge_type]
    train_neg_arr = np.array(dl.get_valid_neg(edge_types)[edge_type])
    shift_i, shift_j = get_shifts(dl, edge_type)
    valid_neg_head = train_neg_arr[0] - shift_i
    valid_neg_tail = train_neg_arr[1] - shift_j
    return valid_neg_head, valid_neg_tail
    
