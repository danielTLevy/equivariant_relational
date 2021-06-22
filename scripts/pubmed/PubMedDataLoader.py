#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from src.DataSchema import DataSchema, Entity, Relation, SparseMatrixData
from src.SparseMatrix import SparseMatrix
import torch
import numpy as np
from scripts.DataLoader import DataLoader

DATA_FILE_DIR = './data/PubMed/'
NODE_FILE_STR = DATA_FILE_DIR + 'node.dat'
LABEL_FILE_STR = DATA_FILE_DIR + 'label.dat'
LINK_FILE_STR = DATA_FILE_DIR + 'link.dat'
TARGET_NODE_TYPE = 1
TARGET_REL_ID = 2

ENTITY_NAMES = ['GENE', 'DISEASE', 'CHEMICAL', 'SPECIES']
RELATION_IDX = {0: (0, 0),
                1: (0, 1),
                2: (1, 1),
                3: (2, 0),
                4: (2, 1),
                5: (2, 2),
                6: (2, 3),
                7: (3, 0),
                8: (3, 1),
                9: (3, 3),
                }
ENTITY_N_INSTANCES = {
                0: 13561,
                1: 20163,
                2: 26522,
                3: 2863
                }

class PubMedData(DataLoader):
    def __init__(self, use_node_attrs=True):
        entities = [Entity(entity_id, n_instances)
                        for entity_id, n_instances
                        in ENTITY_N_INSTANCES.items()]
        relations = [Relation(rel_id, [entities[entity_i], entities[entity_j]])
                        for rel_id, (entity_i, entity_j)
                        in RELATION_IDX.items()]
        if use_node_attrs:
            for entity_id in ENTITY_N_INSTANCES.keys():
                rel = Relation(10 + entity_id, [entities[entity_id], entities[entity_id]],
                               is_set=True)
                relations.append(rel)
        self.schema = DataSchema(entities, relations)

        self.node_id_to_idx = {ent_i: {} for ent_i in range(len(entities))}
        with open(NODE_FILE_STR, 'r') as node_file:
            lines = node_file.readlines()
            node_counter = {ent_i: 0 for ent_i in range(len(entities))}
            for line in lines:
                node_id, node_name, node_type, values = line.rstrip().split('\t')
                node_id = int(node_id)
                node_type = int(node_type)
                node_idx = node_counter[node_type]
                self.node_id_to_idx[node_type][node_id] = node_idx
                node_counter[node_type] += 1
        target_node_id_to_idx = self.node_id_to_idx[TARGET_NODE_TYPE]
        self.target_node_idx_to_id = {idx: id for id, idx
                                      in target_node_id_to_idx.items()}

        raw_data_indices = {rel_id: [] for rel_id in range(len(relations))}
        raw_data_values = {rel_id: [] for rel_id in range(len(relations))}
        if use_node_attrs:
            with open(NODE_FILE_STR, 'r') as node_file:
                lines = node_file.readlines()
                for line in lines:
                    node_id, node_name, node_type, values = line.rstrip().split('\t')
                    node_type = int(node_type)
                    node_id = self.node_id_to_idx[node_type][int(node_id)]
                    values  = list(map(float, values.split(',')))
                    raw_data_indices[10 + node_type].append([node_id, node_id])
                    raw_data_values[10 + node_type].append(values)

        with open(LINK_FILE_STR, 'r') as link_file:
            lines = link_file.readlines()
            for line in lines:
                node_i, node_j, rel_num, val = line.rstrip().split('\t')
                rel_num = int(rel_num)
                node_i_type, node_j_type = RELATION_IDX[rel_num]
                node_i = self.node_id_to_idx[node_i_type][int(node_i)]
                node_j = self.node_id_to_idx[node_j_type][int(node_j)]
                val = float(val)
                raw_data_indices[rel_num].append([node_i, node_j])
                raw_data_values[rel_num].append([val])

        self.data = SparseMatrixData(self.schema)
        for rel in relations:
            indices = torch.LongTensor(raw_data_indices[rel.id]).T
            values = torch.Tensor(raw_data_values[rel.id])
            n = rel.entities[0].n_instances
            m = rel.entities[1].n_instances
            n_channels = values.shape[1]
            data_matrix = SparseMatrix(
                    indices = indices,
                    values = values,
                    shape = np.array([n, m, n_channels]),
                    is_set = rel.is_set
                    )
            del raw_data_indices[rel.id]
            del raw_data_values[rel.id]
            self.data[rel.id] = data_matrix

    def get_node_classification_data(self):
        entities = self.schema.entities


        self.schema_out = DataSchema([entities[TARGET_NODE_TYPE]],
                                [Relation(0, 
                                          [entities[TARGET_NODE_TYPE],
                                           entities[TARGET_NODE_TYPE]],
                                           is_set=True)])
        target_indices = []
        targets = []
        with open(LABEL_FILE_STR, 'r') as label_file:
            lines = label_file.readlines()
            for line in lines:
                node_id, node_name, node_type, node_label = line.rstrip().split('\t')
                node_type = int(node_type)
                node_id = self.node_id_to_idx[node_type][int(node_id)]
                node_label  = int(node_label)
                target_indices.append(node_id)
                targets.append(node_label)
    
        self.target_indices = torch.LongTensor(target_indices)
        self.targets = torch.LongTensor(targets)
        self.n_outputs = self.schema.entities[TARGET_NODE_TYPE].n_instances
        self.data_target = SparseMatrixData(self.schema_out)

    def get_link_prediction_data():
        pass