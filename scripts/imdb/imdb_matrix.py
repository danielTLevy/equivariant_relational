# -*- coding: utf-8 -*-

'''
Source of IMDB dataset, as well as dataset preparation procedures:
https://github.com/YuxiangRen/Heterogeneous-Deep-Graph-Infomax/tree/master/data/IMDB
'''
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
import scipy.sparse as sp
import csv
import random
import pdb
import argparse
import wandb
import pickle

def get_hyperparams(argv):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.set_defaults(dataset='IMDB')
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
    parser.set_defaults(wandb_log_run=False)

    args, argv = parser.parse_known_args(argv)
    args.layers  = [int(x) for x in args.layers]
    args.fc_layers = [int(x) for x in args.fc_layers]

    return args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_file_dir = './data/imdb_het/'
entity_names = ['movie', 'actor', 'director', 'keyword']
relation_names = ['movie_actor', 'movie_director', 'movie_keyword', 'movie_feature']


#%%
def set_seed(seed):
    random.seed(seed, version=2)
    np.random.seed(random.randint(0, 2**32))
    torch.manual_seed(random.randint(0, 2**32))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    argv = sys.argv[1:]
    args = get_hyperparams(argv)
    print(args)
    set_seed(args.seed)

    raw_data = {}
    for rel_name in relation_names:
        file_name = data_file_dir + rel_name + '_adj.pickle'
        with open(file_name, 'rb') as rel_file:
            raw_data[rel_name] = pickle.load(rel_file)
    
    def preprocess_features(features):
        """Row-normalize feature matrix and convert to tuple representation"""
        features = sp.csr_matrix(features, dtype=np.float32) 
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return torch.Tensor(features.todense())
    
    ent_movie = Entity(0, raw_data['movie_feature'].shape[0])
    ent_actor = Entity(1, raw_data['movie_actor'].shape[1])
    ent_director = Entity(2, raw_data['movie_director'].shape[1])
    ent_keyword = Entity(3, raw_data['movie_keyword'].shape[1])
    entities = [ent_movie, ent_actor, ent_director, ent_keyword]
    
    relations = []
    rel_movie_actor = Relation(0, [ent_movie, ent_actor])
    rel_movie_director = Relation(1, [ent_movie, ent_director])
    rel_movie_keyword = Relation(2, [ent_movie, ent_keyword])
    rel_movie_feature = Relation(3, [ent_movie, ent_movie], is_set=True)
    relations = [rel_movie_actor, rel_movie_director, rel_movie_keyword, rel_movie_feature]
    
    schema = DataSchema(entities, relations)
    schema_out = DataSchema([ent_movie], [Relation(0, [ent_movie, ent_movie], is_set=True)])
    
    data = SparseMatrixData(schema)
    for rel_i, rel_name in enumerate(relation_names):
        if rel_name == 'movie_feature':
            values = preprocess_features(raw_data[rel_name])
            data[rel_i] = SparseMatrix.from_embed_diag(values)
        else:
            data[rel_i] = SparseMatrix.from_scipy_sparse(raw_data[rel_name])
    data = data.to(device)
    indices_identity, indices_transpose = data.calculate_indices()
    input_channels = {rel.id: data[rel.id].n_channels for rel in relations}
    data_target = Data(schema_out)
    n_movies = ent_movie.n_instances
    labels = []
    with open(data_file_dir + 'index_label.txt', 'r') as label_file:
        lines = label_file.readlines()
        for line in lines:
            label = line.rstrip().split(',')[1]
            labels.append(int(label))
    labels = torch.LongTensor(labels).to(device) - min(labels)
    
    shuffled_indices = random.sample(range(n_movies), n_movies)
    val_start = 0
    test_start = int(args.val_pct * (n_movies/100.))
    train_start =  test_start + int(args.test_pct * (n_movies/100.))

    val_indices  = sorted(shuffled_indices[val_start:test_start])
    test_indices  = sorted(shuffled_indices[test_start:train_start])
    train_indices = sorted(shuffled_indices[train_start:])

    all_val_indices = sorted(val_indices + train_indices)
    all_test_indices = sorted(val_indices + test_indices + train_indices)
    train_targets = labels[train_indices]
    val_targets = labels[val_indices]
    test_targets = labels[test_indices]
    
    n_output_classes = len(labels.unique())
    data_target[0] = SparseMatrix(
            indices = torch.arange(n_movies, dtype=torch.int64).repeat(2,1),
            values=torch.zeros([n_movies, n_output_classes]),
            shape=(n_movies, n_movies, n_output_classes))
    data_target = data_target.to(device)
    #%%
    net = SparseMatrixEntityPredictor(schema, input_channels,
                                         layers = args.layers,
                                         fc_layers=args.fc_layers,
                                         activation=eval('nn.%s()' % args.act_fn),
                                         final_activation = nn.Identity(),
                                         target_entities=schema_out.entities,
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
        return F.cross_entropy(data_pred, data_true)

    def acc_fcn(values, target):
        return ((values.argmax(1) == target).sum() / len(target)).item()

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
            wandb_log = {'Train Loss': train_loss.item(), 'Train Accuracy': acc}
            if epoch % args.val_every == 0:
                net.eval()
                data_out_val = net(data, indices_identity, indices_transpose, data_target).squeeze()
                data_out_val_values = data_out_val[val_indices]
                val_loss = loss_fcn(data_out_val_values, val_targets)
                val_acc = acc_fcn(data_out_val_values, val_targets)
                print("\nVal Acc: {:.3f} Val Loss: {:.3f}".format(val_acc, val_loss))
                wandb_log.update({'Val Loss': val_loss.item(), 'Val Accuracy': val_acc})
                if val_acc > val_acc_best:
                    val_acc_best = val_acc
                    print("New best")
                if not args.no_scheduler:
                    sched.step(val_loss)
            if epoch % args.wandb_log_loss_freq == 0:
                if args.wandb_log_run:
                    wandb.log(wandb_log)
