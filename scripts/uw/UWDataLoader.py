#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from src.DataSchema import DataSchema, Entity, Relation, SparseMatrixData, Data
from src.SparseMatrix import SparseMatrix
from src.EquivariantNetwork import EquivariantNetwork
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import csv
import random
import pdb
import wandb
from collections import OrderedDict

from scripts.DataLoader import DataLoader
from src.SparseMatrix import SparseMatrix

csv_file_str = './data/uw_cse/uw_std_{}.csv'
entity_names = ['person', 'course']
relation_names = ['person', 'course', 'advisedBy', 'taughtBy']

schema_dict = {
        'person': OrderedDict({
                'p_id': 'id', # ID
                'professor': 'binary', 
                'student': 'binary',
                'hasPosition': 'categorical',
                'inPhase': 'categorical',
                'yearsInProgram': 'categorical'
                }),
        'course': OrderedDict({
                'course_id': 'id',
                'courseLevel': 'categorical'
                }),
        'advisedBy': OrderedDict({ 
                'p_id': 'id', # Student
                'p_id_dummy': 'id' # Professor
                }),
        'taughtBy': OrderedDict({
                'course_id': 'id', # Course
                'p_id': 'id'
                })
        }


class UWData(DataLoader):
    def binary_to_tensor(self, values_list):
        assert len(np.unique(values_list)) == 2
        value = values_list[0]
        return torch.Tensor(np.array(values_list) == value).unsqueeze(1)

    def ordinal_to_tensor(self, values_list):
        values_int_list = list(map(int, values_list))
        return torch.Tensor(np.array(values_int_list)).unsqueeze(1)

    def categorical_to_tensor(self, values_list):
        values_array = np.array(values_list)
        categories = np.unique(values_array)
        one_hot = torch.zeros(len(values_list), len(categories))
        for i, category in enumerate(categories):
            one_hot[:, i] = torch.Tensor(values_array == category)
        return one_hot

    def set_relation_to_matrix(self, relation, typedict, raw_vals):
        assert relation.entities[0] == relation.entities[1]
        assert relation.is_set
        n_instances = relation.entities[0].n_instances
        tensor_list = []
        for key, val in typedict.items():
            if key == self.TARGET_KEY:
                continue
            if val == 'id':
                continue
            elif val == 'ordinal':
                func = self.ordinal_to_tensor
            elif val == 'categorical':
                func = self.categorical_to_tensor
            elif val == 'binary':
                func = self.binary_to_tensor
            tensor_list.append(func(raw_vals[key]))
    
        if len(tensor_list) != 0:
            values = torch.cat(tensor_list, 1)
        else:
            values = torch.ones(n_instances, 1)
        indices = torch.arange(n_instances).repeat(2,1)
        return SparseMatrix(
                indices = indices,
                values = values,
                shape = (n_instances, n_instances, values.shape[1]))

    def id_to_idx(self, ids_list):
        return {ids_list[i]: i  for i in range(len(ids_list))}

    def ent_name_to_id_name(self, ent_name):
        return ent_name[:-1] + 'id'

    def binary_relation_to_matrix(self, relation, typedict, raw_vals,
                                  ent_id_to_idx_dict,  ent_n_id_str, ent_m_id_str):
        assert not relation.is_set
        ent_n = relation.entities[0]
        ent_n_name = entity_names[ent_n.id]
        ent_m = relation.entities[1]
        ent_m_name = entity_names[ent_m.id]
        instances_n = ent_n.n_instances
        instances_m = ent_m.n_instances
        tensor_list = []
        for key, val in typedict.items():
            if val == 'id':
                continue
            elif val == 'ordinal':
                func = self.ordinal_to_tensor
            elif val == 'categorical':
                func = self.categorical_to_tensor
            elif val == 'binary':
                func = self.binary_to_tensor
            tensor_list.append(func(raw_vals[key]))
        n_ids = raw_vals[ent_n_id_str]
        m_ids = raw_vals[ent_m_id_str]
        if len(tensor_list) != 0:
            values = torch.cat(tensor_list, 1)
        else:
            values = torch.ones(len(n_ids), 1)
        indices_n = torch.LongTensor([
                ent_id_to_idx_dict[ent_n_name][ent_i] for ent_i in n_ids
                ])
        indices_m = torch.LongTensor([
                ent_id_to_idx_dict[ent_m_name][ent_i] for ent_i in m_ids
                ])
    
        return SparseMatrix(
                indices = torch.stack((indices_n, indices_m)),
                values = values,
                shape = (instances_n, instances_m, values.shape[1]))

class UWAdvisorData(UWData):
    '''
    DataLoader for predicting the advisedBy relation of the UW-CSE dataset
    '''
    TARGET_RELATION = 'advisedBy'
    TARGET_KEY = 'p_id_dummy'
    
    def __init__(self):
        self.target_relation = 'advisedBy'
        
        data_raw = {rel_name: {key: list() for key in schema_dict[rel_name].keys()}
                    for rel_name in schema_dict.keys()}
    
        for relation_name in relation_names:
            with open(csv_file_str.format(relation_name)) as file:
                reader = csv.reader(file)
                keys = schema_dict[relation_name].keys()
                for cols in reader:
                    for key, col in zip(keys, cols):
                        data_raw[relation_name][key].append(col)

        ent_person = Entity(0, len(data_raw['person']['p_id']))
        ent_course = Entity(1, len(data_raw['course']['course_id']))
        entities = [ent_person, ent_course]

        rel_person_matrix = Relation(0, [ent_person, ent_person], is_set=True)
        rel_person = Relation(0, [ent_person])
        rel_course_matrix = Relation(1, [ent_course, ent_course], is_set=True)
        rel_course = Relation(1, [ent_course])
        rel_advisedBy = Relation(2, [ent_person, ent_person])
        rel_taughtBy = Relation(3, [ent_course, ent_person])
        relations_matrix = [rel_person_matrix, rel_course_matrix, rel_advisedBy, rel_taughtBy]
        relations = [rel_person, rel_course, rel_taughtBy]

        self.target_rel_id = 2
        self.schema = DataSchema(entities, relations)
        schema_matrix = DataSchema(entities, relations_matrix)
        matrix_data = SparseMatrixData(schema_matrix)

        ent_id_to_idx_dict = {'person': self.id_to_idx(data_raw['person']['p_id']),
                              'course': self.id_to_idx(data_raw['course']['course_id'])}

        for relation in relations_matrix:
            relation_name = relation_names[relation.id]
            print(relation_name)
            if relation.is_set:
                data_matrix = self.set_relation_to_matrix(relation, schema_dict[relation_name],
                                                    data_raw[relation_name])
            else:
                if relation_name == 'advisedBy':
                    ent_n_id_str = 'p_id'
                    ent_m_id_str = 'p_id_dummy'
                elif relation_name == 'taughtBy':
                    ent_n_id_str = 'course_id'
                    ent_m_id_str = 'p_id'
                data_matrix = self.binary_relation_to_matrix(relation, schema_dict[relation_name],
                                                    data_raw[relation_name], ent_id_to_idx_dict,
                                                    ent_n_id_str, ent_m_id_str)
            matrix_data[relation.id] = data_matrix

        rel_out = Relation(2, [ent_person, ent_person])
        self.schema_out = DataSchema([ent_person], [rel_out])

        self.output_dim = 1
        data = Data(self.schema)
        for rel_matrix in schema_matrix.relations:
            for rel in self.schema.relations:
                if rel_matrix.id == rel.id:
                    data_matrix = matrix_data[rel_matrix.id]
                    if rel_matrix.is_set:
                        dense_data = torch.diagonal(data_matrix.to_dense(), 0, 1, 2).unsqueeze(0)
                    else:
                        dense_data = data_matrix.to_dense().unsqueeze(0)
                    data[rel.id] = dense_data
        self.data = data

        self.target = matrix_data[self.target_rel_id].to_dense().squeeze()


class UWPhaseData(UWData):
    '''
    DataLoader for predicting the inPhase field of a person in the UW-CSE
    dataset
    '''
    TARGET_RELATION = 'person'
    TARGET_KEY = 'inPhase'
    def __init__(self):
        data_raw = {rel_name: {key: list() for key in schema_dict[rel_name].keys()}
                    for rel_name in schema_dict.keys()}
    
        for relation_name in relation_names:
            with open(csv_file_str.format(relation_name)) as file:
                reader = csv.reader(file)
                keys = schema_dict[relation_name].keys()
                for cols in reader:
                    for key, col in zip(keys, cols):
                        data_raw[relation_name][key].append(col)
    
        ent_person = Entity(0, len(data_raw['person']['p_id']))
        ent_course = Entity(1, len(data_raw['course']['course_id']))
        entities = [ent_person, ent_course]
    
        rel_person = Relation(0, [ent_person, ent_person], is_set=True)
        rel_course = Relation(1, [ent_course, ent_course], is_set=True)
        rel_advisedBy = Relation(2, [ent_person, ent_person])
        rel_taughtBy = Relation(3, [ent_course, ent_person])
        relations = [rel_person, rel_course, rel_advisedBy, rel_taughtBy]
    
        self.schema = DataSchema(entities, relations)
        self.data = SparseMatrixData(self.schema)

        ent_id_to_idx_dict = {'person': self.id_to_idx(data_raw['person']['p_id']),
                              'course': self.id_to_idx(data_raw['course']['course_id'])}

        for relation in relations:
            relation_name = relation_names[relation.id]
            print(relation_name)
            if relation.is_set:
                data_matrix = self.set_relation_to_matrix(relation, schema_dict[relation_name],
                                                    data_raw[relation_name])
            else:
                if relation_name == 'advisedBy':
                    ent_n_id_str = 'p_id'
                    ent_m_id_str = 'p_id_dummy'
                elif relation_name == 'taughtBy':
                    ent_n_id_str = 'course_id'
                    ent_m_id_str = 'p_id'
                data_matrix = self.binary_relation_to_matrix(relation, schema_dict[relation_name],
                                                    data_raw[relation_name], ent_id_to_idx_dict,
                                                    ent_n_id_str, ent_m_id_str)
            self.data[relation.id] = data_matrix

        self.target = self.get_targets(data_raw[self.TARGET_RELATION][self.TARGET_KEY],
                             schema_dict[self.TARGET_RELATION][self.TARGET_KEY])
        self.target_rel_id = 0
        rel_out = Relation(self.target_rel_id, [ent_person, ent_person], is_set=True)
        self.schema_out = DataSchema([ent_person], [rel_out])
        self.data_target = Data(self.schema_out)
        n_output_classes = len(np.unique(data_raw[self.TARGET_RELATION][self.TARGET_KEY]))
        self.output_dim = n_output_classes
        n_person = ent_person.n_instances
        self.data_target[self.target_rel_id] = SparseMatrix(
                            indices=torch.arange(n_person, dtype=torch.int64).repeat(2,1),
                            values=torch.zeros([n_person, n_output_classes]),
                            shape=(n_person, n_person, n_output_classes))

    def get_targets(self, target_vals, target_type):
        if target_type == 'ordinal':
            return torch.Tensor(self.ordinal_to_tensor(target_vals)).squeeze()
        elif target_type == 'categorical':
            return torch.Tensor(self.categorical_to_tensor(target_vals)).argmax(1)
        elif target_type == 'binary':
            return torch.Tensor(self.binary_to_tensor(target_vals)).squeeze()
