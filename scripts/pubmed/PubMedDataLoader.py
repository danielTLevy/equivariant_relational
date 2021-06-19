#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from src.DataSchema import DataSchema, Entity, Relation, SparseMatrixData
from src.SparseMatrix import SparseMatrix
import torch
import numpy as np
from scripts.DataLoader import DataLoader

data_file_dir = './data/PubMed/'
entity_names = ['GENE', 'DISEASE', 'CHEMICAL', 'SPECIES']
relation_idx = {0: (0, 0),
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

class PubMedData(DataLoader):
    def __init__(self, node_labels):
        TARGET_NODE_TYPE = 1
        entities = [
                Entity(0, 13561),
                Entity(1, 20163),
                Entity(2, 26522),
                Entity(3, 2863)
                ]
        relations = []
        for rel_id in range(10):
            rel = Relation(rel_id, [entities[relation_idx[rel_id][0]],
                                    entities[relation_idx[rel_id][1]]])
            relations.append(rel)
        if node_labels:
            for entity_id in range(0, 4):
                rel = Relation(10 + entity_id, [entities[entity_id], entities[entity_id]],
                               is_set=True)
                relations.append(rel)
        self.schema = DataSchema(entities, relations)

        node_file_str = data_file_dir + 'node.dat'
        node_id_to_idx = {ent_i: {} for ent_i in range(len(entities))}
        self.target_node_idx_to_id = {}
        with open(node_file_str, 'r') as node_file:
            lines = node_file.readlines()
            node_counter = {ent_i: 0 for ent_i in range(len(entities))}
            for line in lines:
                node_id, node_name, node_type, values = line.rstrip().split('\t')
                node_id = int(node_id)
                node_type = int(node_type)
                node_idx = node_counter[node_type]
                node_id_to_idx[node_type][node_id] = node_idx
                if node_type == TARGET_NODE_TYPE:
                    self.target_node_idx_to_id[node_idx] = node_id
                node_counter[node_type] += 1

        raw_data_indices = {rel_id: [] for rel_id in range(len(relations))}
        raw_data_values = {rel_id: [] for rel_id in range(len(relations))}

        if node_labels:
            with open(node_file_str, 'r') as node_file:
                lines = node_file.readlines()
                for line in lines:
                    node_id, node_name, node_type, values = line.rstrip().split('\t')
                    node_type = int(node_type)
                    node_id = node_id_to_idx[node_type][int(node_id)]
                    values  = list(map(float, values.split(',')))
                    raw_data_indices[10 + node_type].append([node_id, node_id])
                    raw_data_values[10 + node_type].append(values)

        link_file_str = data_file_dir + 'link.dat'
        with open(link_file_str, 'r') as link_file:
            lines = link_file.readlines()
            for line in lines:
                node_i, node_j, rel_num, val = line.rstrip().split('\t')
                rel_num = int(rel_num)
                node_i_type, node_j_type = relation_idx[rel_num]
                node_i = node_id_to_idx[node_i_type][int(node_i)]
                node_j = node_id_to_idx[node_j_type][int(node_j)]
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

        self.schema_out = DataSchema([entities[TARGET_NODE_TYPE]],
                                [Relation(0, 
                                          [entities[TARGET_NODE_TYPE],
                                           entities[TARGET_NODE_TYPE]],
                                           is_set=True)])
        label_file_str = data_file_dir + 'label.dat'
        target_indices = []
        targets = []
        with open(label_file_str, 'r') as label_file:
            lines = label_file.readlines()
            for line in lines:
                node_id, node_name, node_type, node_label = line.rstrip().split('\t')
                node_type = int(node_type)
                node_id = node_id_to_idx[node_type][int(node_id)]
                node_label  = int(node_label)
                target_indices.append(node_id)
                targets.append(node_label)
    
        self.target_indices = torch.LongTensor(target_indices)
        self.targets = torch.LongTensor(targets)
        self.n_outputs = entities[TARGET_NODE_TYPE].n_instances
        self.data_target = SparseMatrixData(self.schema_out)
