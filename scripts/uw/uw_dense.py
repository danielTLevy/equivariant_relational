# -*- coding: utf-8 -*-
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
import argparse
import wandb
from collections import OrderedDict

def get_hyperparams(argv):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.set_defaults(dataset='UW_CSE')
    parser.add_argument('--checkpoint_path', type=str, default='cora_matrix.pt')
    parser.add_argument('--source_layers', type=int, nargs='*', default=['64']*4,
                        help='Number of channels for equivariant layers with source schema')
    parser.add_argument('--target_layers', type=str, nargs='*', default=['32']*1,
                        help='Number of channels for equivariant layers with target schema')
    parser.add_argument('--l2_decay', type=float, default=0)
    parser.add_argument('--dropout_rate', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--act_fn', type=str, default='ReLU')
    parser.add_argument('--no_scheduler', action='store_true', default=False)
    parser.add_argument('--sched_factor', type=float, default=0.5)
    parser.add_argument('--sched_patience', type=float, default=10)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--val_every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--norm',  dest='norm', action='store_true', default=True)
    parser.add_argument('--no_norm', dest='norm', action='store_false', default=True)
    parser.set_defaults(norm=True)
    parser.add_argument('--norm_affine', action='store_true')
    parser.set_defaults(norm_affine = False)
    parser.add_argument('--pool_op', type=str, default='mean')
    parser.add_argument('--neg_data', type=float, default=0.,
                        help='Ratio of random data samples to positive. \
                              When sparse, this is similar to number of negative samples')
    parser.add_argument('--training_data', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--val_pct', type=float, default=10.)
    parser.add_argument('--test_pct', type=float, default=10.)
    parser.add_argument('--semi_supervised', action='store_true', help='switch to low-label regime')
    parser.set_defaults(semi_supervised=False)
    parser.add_argument('--wandb_log_param_freq', type=int, default=250)
    parser.add_argument('--wandb_log_loss_freq', type=int, default=1)
    parser.add_argument('--wandb_log_run', dest='wandb_log_run', action='store_true',
                        help='Log this run in wandb')
    parser.add_argument('--wandb_no_log_run', dest='wandb_log_run', action='store_false',
                        help='Do not log this run in wandb')
    parser.set_defaults(wandb_log_run=False)

    args, argv = parser.parse_known_args(argv)
    args.source_layers  = [int(x) for x in args.source_layers]
    args.target_layers = [int(x) for x in args.target_layers]

    return args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
TARGET_RELATION = 'advisedBy'
TARGET_KEY = 'p_id_dummy'


def binary_to_tensor(values_list):
    assert len(np.unique(values_list)) == 2
    value = values_list[0]
    return torch.Tensor(np.array(values_list) == value).unsqueeze(1)

def ordinal_to_tensor(values_list):
    values_int_list = list(map(int, values_list))
    return torch.Tensor(np.array(values_int_list)).unsqueeze(1)
    
def categorical_to_tensor(values_list):
    values_array = np.array(values_list)
    categories = np.unique(values_array)
    one_hot = torch.zeros(len(values_list), len(categories))
    for i, category in enumerate(categories):
        one_hot[:, i] = torch.Tensor(values_array == category)
    return one_hot

def set_relation_to_matrix(relation, typedict, raw_vals):
    assert relation.entities[0] == relation.entities[1]
    assert relation.is_set
    n_instances = relation.entities[0].n_instances
    tensor_list = []
    for key, val in typedict.items():
        if key == TARGET_KEY:
            continue
        if val == 'id':
            continue
        elif val == 'ordinal':
            func = ordinal_to_tensor
        elif val == 'categorical':
            func = categorical_to_tensor
        elif val == 'binary':
            func = binary_to_tensor
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

def id_to_idx(ids_list):
    return {ids_list[i]: i  for i in range(len(ids_list))}

def ent_name_to_id_name(ent_name):
    return ent_name[:-1] + 'id'

def binary_relation_to_matrix(relation, typedict, raw_vals, ent_id_to_idx_dict,
                              ent_n_id_str, ent_m_id_str):
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
            func = ordinal_to_tensor
        elif val == 'categorical':
            func = categorical_to_tensor
        elif val == 'binary':
            func = binary_to_tensor
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


def get_targets(target_vals, target_type):
    if target_type == 'ordinal':
        return torch.Tensor(ordinal_to_tensor(target_vals)).squeeze()
    elif target_type == 'categorical':
        return torch.Tensor(categorical_to_tensor(target_vals)).argmax(1)
    elif target_type == 'binary':
        return torch.Tensor(binary_to_tensor(target_vals)).squeeze()
    

#%%
if __name__ == '__main__':
    argv = sys.argv[1:]
    args = get_hyperparams(argv)
    print(args)

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
    TARGET_RELATION_ID = 2
    rel_taughtBy = Relation(3, [ent_course, ent_person])
    relations_matrix = [rel_person_matrix, rel_course_matrix, rel_advisedBy, rel_taughtBy]
    relations = [rel_person, rel_course, rel_taughtBy]

    schema = DataSchema(entities, relations)
    schema_matrix = DataSchema(entities, relations_matrix)
    matrix_data = SparseMatrixData(schema_matrix)
    
    schema_out = DataSchema(entities, [rel_advisedBy])


    ent_id_to_idx_dict = {'person': id_to_idx(data_raw['person']['p_id']),
                          'course': id_to_idx(data_raw['course']['course_id'])}

    for relation in relations_matrix:
        relation_name = relation_names[relation.id]
        print(relation_name)
        if relation.is_set:
            data_matrix = set_relation_to_matrix(relation, schema_dict[relation_name],
                                                data_raw[relation_name])
        else:
            if relation_name == 'advisedBy':
                ent_n_id_str = 'p_id'
                ent_m_id_str = 'p_id_dummy'
            elif relation_name == 'taughtBy':
                ent_n_id_str = 'course_id'
                ent_m_id_str = 'p_id'
            data_matrix = binary_relation_to_matrix(relation, schema_dict[relation_name],
                                                data_raw[relation_name], ent_id_to_idx_dict,
                                                ent_n_id_str, ent_m_id_str)
        matrix_data[relation.id] = data_matrix


    #target = get_targets(data_raw[TARGET_RELATION][TARGET_KEY],
    #                     schema_dict[TARGET_RELATION][TARGET_KEY]).to(device)

    rel_out = Relation(2, [ent_person, ent_person])
    schema_out = DataSchema([ent_person], [rel_out])
    data_target = Data(schema_out)
    #n_output_classes = len(np.unique(data_raw[TARGET_RELATION][TARGET_KEY]))
    n_output_classes = 1
    n_person = ent_person.n_instances
    data = Data(schema)
    for rel_matrix in schema_matrix.relations:
        for rel in schema.relations:
            if rel_matrix.id == rel.id:
                data_matrix = matrix_data[rel_matrix.id]
                if rel_matrix.is_set:
                    dense_data = torch.diagonal(data_matrix.to_dense(), 0, 1, 2).unsqueeze(0)
                else:
                    dense_data = data_matrix.to_dense().unsqueeze(0)
                data[rel.id] = dense_data
    data = data.to(device)

    target = matrix_data[TARGET_RELATION_ID].to_dense().squeeze().to(device)
    #%%
    input_channels = {rel.id: data[rel.id].shape[1] for rel in relations}

    shuffled_indices = random.sample(range(n_person), n_person)
    val_start = 0
    test_start = int(args.val_pct * (n_person/100.))
    train_start =  test_start + int(args.test_pct * (n_person/100.))

    val_indices  = sorted(shuffled_indices[val_start:test_start])
    test_indices  = sorted(shuffled_indices[test_start:train_start])
    train_indices = sorted(shuffled_indices[train_start:])

    all_val_indices = sorted(val_indices + train_indices)
    all_test_indices = sorted(val_indices + test_indices + train_indices)
    #train_targets = target[train_indices]
    #val_targets = target[val_indices]
    #%%
    net = EquivariantNetwork(schema, input_channels,
                                         source_layers=args.source_layers,
                                         target_layers=args.target_layers,
                                         activation=eval('nn.%s()' % args.act_fn),
                                         final_activation = nn.Sigmoid(),
                                         schema_out=schema_out,
                                         dropout=args.dropout_rate,
                                         output_dim=n_output_classes,
                                         norm=args.norm,
                                         pool_op=args.pool_op,
                                         norm_affine=args.norm_affine)

    net = net.to(device)
    opt = eval('optim.%s' % args.optimizer)(net.parameters(), lr=args.learning_rate, weight_decay=args.l2_decay)

    if not args.no_scheduler:
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                     factor=args.sched_factor,
                                                     patience=args.sched_patience,
                                                     verbose=True)

    #%%
    if args.wandb_log_run:
        wandb.init(config=args,
            project="EquivariantRelational",
            entity='danieltlevy',
            settings=wandb.Settings(start_method='fork'))
        wandb.watch(net, log='all', log_freq=args.wandb_log_param_freq)

    progress = tqdm(range(args.num_epochs), desc="Epoch 0", position=0, leave=True)

    def loss_fcn(data_pred, data_true):
        return F.binary_cross_entropy(data_pred, data_true)

    def acc_fcn(values, target):
        return (((values > 0.5) == target).sum() / target.numel()).item()

    val_acc_best = 0
    for epoch in progress:
        net.train()
        opt.zero_grad()
        data_out = net(data)[rel_advisedBy.id].squeeze()
        train_loss = loss_fcn(data_out, target)
        train_loss.backward()
        opt.step()
        with torch.no_grad():
            acc = acc_fcn(data_out, target)
            progress.set_description(f"Epoch {epoch}")
            progress.set_postfix(loss=train_loss.item(), train_acc=acc)
