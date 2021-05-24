# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import sys
from src.DataSchema import DataSchema, Entity, Relation, SparseMatrixData, Data
from src.SparseMatrix import SparseMatrix
from src.EquivariantNetwork import SparseMatrixEntityPredictor
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
    parser.add_argument('--checkpoint_path', type=str, default='cora_matrix.pt')
    parser.add_argument('--layers', type=int, nargs='*', default=['64']*4,
                        help='Number of channels for equivariant layers')
    parser.add_argument('--fc_layers', type=str, nargs='*', default=[],
                        help='Fully connected layers for target embeddings')
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
    parser.set_defaults(wandb_log_run=True)


    args, argv = parser.parse_known_args(argv)
    args.layers  = [int(x) for x in args.layers]
    args.fc_layers = [int(x) for x in args.fc_layers]

    return args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv_file_str = './data/movielens/movielens_{}.csv'
entity_names = ['actors', 'movies', 'directors', 'users']
relation_names = ['actors', 'movies', 'directors', 'users', 'movies2actors', 
                  'movies2directors', 'u2base']

schema_dict = {
        'actors': OrderedDict({
                'actorid': 'id', # ID
                'a_gender': 'binary', # two values
                'a_quality': 'ordinal' # 6 values, ordinal
                }),
        'movies': OrderedDict({
                'movieid': 'id', #ID
                'year': 'ordinal', # 4 values, ordinal
                'isEnglish': 'binary', # two values
                'country': 'categorical', #  4 values
                'runningtime': 'ordinal' # 4 values, ordinal
                }),
        'directors': OrderedDict({ 
                'directorid': 'id', # ID
                'd_quality': 'ordinal', # 6 values, ordinal
                'avg_revenue': 'ordinal' # 5 values, ordinal
                }),
        'users': OrderedDict({
                'userid': 'id', # ID
                'age': 'ordinal', # 6 non-consecutive values, ordinal
                'u_gender': 'binary', # 2 values
                'occupation': 'categorical' # 5 values, categorial
                }),
        'movies2actors': OrderedDict({
                'movieid': 'id',
                'actorid': 'id',
                'cast_num': 'categorical' # 5 values, not sure if ordinal
                }),
        'movies2directors': OrderedDict({
                'movieid': 'id',
                'directorid': 'id',
                'genre': 'categorical' # 9 values
                }),
        'u2base': OrderedDict({
                'userid': 'id',
                'movieid': 'id',
                'rating': 'ordinal' # 5 values, ordinal
                })
        }
TARGET_RELATION = 'users'
TARGET_KEY = 'u_gender'


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
    values = torch.cat(tensor_list, 1)
    indices = torch.arange(n_instances).repeat(2,1)
    return SparseMatrix(
            indices = indices,
            values = values,
            shape = (n_instances, n_instances, values.shape[1]))

def id_to_idx(ids_list):
    return {ids_list[i]: i  for i in range(len(ids_list))}

def ent_name_to_id_name(ent_name):
    return ent_name[:-1] + 'id'

def binary_relation_to_matrix(relation, typedict, raw_vals, ent_id_to_idx_dict):
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
    values = torch.cat(tensor_list, 1)
    n_ids = raw_vals[ent_name_to_id_name(ent_n_name)]
    m_ids = raw_vals[ent_name_to_id_name(ent_m_name)]
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
        func = ordinal_to_tensor
    elif target_type == 'categorical':
        func = categorical_to_tensor
    elif target_type == 'binary':
        func = binary_to_tensor
    return torch.Tensor(func(target_vals)).squeeze()

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

    entity_names = ['actors', 'movies', 'directors', 'users']
    rel_names = ['actors', 'movies', 'directors', 'users',
                 'movies2actors', 'movies2directors', 'u2base']
    ent_actors = Entity(0, len(data_raw['actors']['actorid']))
    ent_movies = Entity(1, len(data_raw['movies']['movieid']))
    ent_directors = Entity(2, len(data_raw['directors']['directorid']))
    ent_users = Entity(3, len(data_raw['users']['userid']))
    entities = [ent_actors, ent_movies, ent_directors, ent_users]

    rel_actors = Relation(0, [ent_actors, ent_actors], is_set=True)
    rel_movies = Relation(1, [ent_movies, ent_movies], is_set=True)
    rel_directors = Relation(2, [ent_directors, ent_directors], is_set=True)
    rel_users = Relation(3, [ent_users, ent_users], is_set=True)
    rel_movies2actors = Relation(4, [ent_movies, ent_actors])
    rel_movies2directors = Relation(5, [ent_movies, ent_directors])
    rel_u2base = Relation(6, [ent_users, ent_movies])
    relations = [rel_actors, rel_movies, rel_directors, rel_users,
                 rel_movies2actors, rel_movies2directors, rel_u2base]

    schema = DataSchema(entities, relations)
    data = SparseMatrixData(schema)


    ent_id_to_idx_dict = {ent_name: id_to_idx(data_raw[ent_name][ent_name_to_id_name(ent_name)])
                        for ent_name in entity_names}

    for relation in relations:
        relation_name = rel_names[relation.id]
        print(relation_name)
        if relation.is_set:
            data_matrix = set_relation_to_matrix(relation, schema_dict[relation_name],
                                                data_raw[relation_name])
        else:
            data_matrix = binary_relation_to_matrix(relation, schema_dict[relation_name],
                                                data_raw[relation_name], ent_id_to_idx_dict)
        data[relation.id] = data_matrix


    target = get_targets(data_raw[TARGET_RELATION][TARGET_KEY],
                         schema_dict[TARGET_RELATION][TARGET_KEY]).to(device)

    rel_out = Relation(0, [ent_users, ent_users], is_set=True)
    schema_out = DataSchema([ent_users], [rel_out])
    data_target = Data(schema_out)
    output_classes = 1
    n_users = ent_users.n_instances
    data_target[0] = SparseMatrix(
                        indices=torch.arange(n_users, dtype=torch.int64).repeat(2,1),
                        values=torch.zeros([n_users, output_classes]),
                        shape=(n_users, n_users, output_classes))
    data = data.to(device)
    data_target = data_target.to(device)
    indices_identity, indices_transpose = data.calculate_indices()

    input_channels = {rel.id: data[rel.id].n_channels for rel in relations}

    shuffled_indices = random.sample(range(n_users), n_users)
    val_start = 0
    test_start = int(args.val_pct * (n_users/100.))
    train_start =  test_start + int(args.test_pct * (n_users/100.))

    val_indices  = sorted(shuffled_indices[val_start:test_start])
    test_indices  = sorted(shuffled_indices[test_start:train_start])
    train_indices = sorted(shuffled_indices[train_start:])

    all_val_indices = sorted(val_indices + train_indices)
    all_test_indices = sorted(val_indices + test_indices + train_indices)
    train_targets = target[train_indices]
    val_targets = target[val_indices]
    #%%
    net = SparseMatrixEntityPredictor(schema, input_channels,
                                         layers = args.layers,
                                         fc_layers=args.fc_layers,
                                         activation=eval('nn.%s()' % args.act_fn),
                                         final_activation = nn.Sigmoid(),
                                         target_entities=schema_out.entities,
                                         dropout=args.dropout_rate,
                                         output_dim=output_classes,
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

    progress = tqdm(range(args.num_epochs), desc="Epoch 0", position=0, leave=True)

    loss_fcn = nn.BCELoss().to(device)
    def acc_fcn(values, target):
        return (((values > 0.5) == target).sum() / len(target)).item()

    val_acc_best = 0
    for epoch in progress:
        net.train()
        opt.zero_grad()
        data_out = net(data, indices_identity, indices_transpose, data_target).squeeze()
        data_out_train_values = data_out[train_indices]
        train_loss = loss_fcn(data_out_train_values, train_targets)
        train_loss.backward()
        opt.step()
        with torch.no_grad():
            acc = acc_fcn(data_out_train_values, train_targets)
            progress.set_description(f"Epoch {epoch}")
            progress.set_postfix(loss=train_loss.item(), train_acc=acc)
            if epoch % args.val_every == 0:
                net.eval()
                data_out_val = net(data, indices_identity, indices_transpose, data_target).squeeze()
                data_out_val_values = data_out_val[val_indices]
                val_loss = loss_fcn(data_out_val_values, val_targets)
                val_acc = acc_fcn(data_out_val_values, val_targets)
                print("\nVal Acc: {:.3f} Val Loss: {:.3f}".format(val_acc, val_loss))
                if not args.no_scheduler:
                    sched.step(val_loss)
