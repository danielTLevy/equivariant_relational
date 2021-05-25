# -*- coding: utf-8 -*-

'''
Entities:
    paper: 2708: Each has unique int. Nonconsecutive, so must translate to idx
    word: 1433. Each is like "word125" so just need to convert to 125 to get idx
    class: 7. Must translate text to idx
Relationships:
    cites: 5429, paper to paper
    content: 49216, paper to word
    paper: 2708, paper to class
    Alternatively, could make class a channel attribute for a paper?
    This would involve getting the code working for sets too probably
Goal is to predict class - use 10-fold cross validation.
Use 8/10 for train, 1/10 for val, 1/10 for test – train on val before testing on test
'''
#%%
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

#%%
csv_file_str = './data/cora/{}.csv'

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

def get_data_and_targets(schema, neg_data, data_indices, paper, cites, content):
    n_papers = schema.entities[0].n_instances
    n_words = schema.entities[1].n_instances

    train_targets = torch.LongTensor(paper[1])

    # Randomly fill in values and coalesce to remove duplicates
    n_cites_neg = int(neg_data*cites.shape[1])
    #cites_neg = np.random.choice(data_indices, (2, n_cites_neg))
    cites_neg = np.random.randint(0, n_papers, (2, n_cites_neg))
    cites_matrix = SparseMatrix(
            indices = torch.LongTensor(np.concatenate((cites, cites_neg),axis=1)),
            values = torch.cat((torch.ones(cites.shape[1], 1), torch.zeros(n_cites_neg, 1))),
            shape = (n_papers, n_papers, 1)
            ).coalesce()

    # For each paper, randomly fill in values and coalesce to remove duplicates
    n_content_neg = int(neg_data*content.shape[1])
    #content_neg = np.stack((np.random.choice(data_indices, (n_content_neg,)),
    #                np.random.randint(0, n_words, (n_content_neg,))))
    content_neg = np.stack((np.random.randint(0, n_papers, (n_content_neg,)),
                np.random.randint(0, n_words, (n_content_neg,))))
    content_matrix = SparseMatrix(
            indices = torch.LongTensor(np.concatenate((content, content_neg),axis=1)),
            values = torch.cat((torch.ones(content.shape[1], 1), torch.zeros(n_content_neg, 1))),
            shape = (n_papers, n_words, 1)
            ).coalesce()

    data = SparseMatrixData(schema)

    data[0] = cites_matrix
    data[1] = content_matrix
    return data, train_targets


def set_seed(seed):
    random.seed(seed, version=2)
    np.random.seed(random.randint(0, 2**32))
    torch.manual_seed(random.randint(0, 2**32))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#%%
if __name__ == '__main__':

    argv = sys.argv[1:]
    args = get_hyperparams(argv)
    print(args)
    set_seed(args.seed)
    paper_names = []
    classes = []
    word_names = ['word'+str(i+1) for i in range(1433)]
    with open(csv_file_str.format('paper')) as paperfile:
        reader = csv.reader(paperfile)
        for paper_name, class_name in reader:
            paper_names.append(paper_name)
            classes.append(class_name)
    paper_names = np.array(paper_names)

    class_names = list(np.unique(classes))
    class_name_to_idx = {class_name : i for i, class_name in enumerate(class_names)}
    paper_name_to_idx = {paper_name: i for i, paper_name in enumerate(paper_names)}

    shuffled_indices = random.sample(range(len(paper_names)), len(paper_names))
    val_start = 0
    test_start = int(args.val_pct * (len(paper_names)/100.))
    train_start =  test_start + int(args.test_pct * (len(paper_names)/100.))

    val_indices  = sorted(shuffled_indices[val_start:test_start])
    test_indices  = sorted(shuffled_indices[test_start:train_start])
    train_indices = sorted(shuffled_indices[train_start:])

    all_val_indices = sorted(val_indices + train_indices)
    all_test_indices = sorted(val_indices + test_indices + train_indices)
    val_papers = paper_names[val_indices]
    test_papers = paper_names[test_indices]

    train_paper = []
    val_paper = []
    test_paper = []
    for paper_name, class_name in zip(paper_names, classes):
        paper_idx = paper_name_to_idx[paper_name]
        class_idx = class_name_to_idx[class_name]
        if paper_name in test_papers:
            test_paper.append([paper_idx, class_idx])
        elif paper_name in val_papers:
            val_paper.append(([paper_idx, class_idx]))
        else:
            train_paper.append(([paper_idx, class_idx]))
    train_paper = np.array(train_paper).T
    val_paper = np.array(val_paper).T
    test_paper = np.array(test_paper).T


    train_cites = []
    test_cites = []
    val_cites = []
    with open(csv_file_str.format('cites')) as citesfile:
        reader = csv.reader(citesfile)
        for citer, citee in reader:
            citer_idx = paper_name_to_idx[citer]
            citee_idx = paper_name_to_idx[citee]
            test_cites.append([citer_idx, citee_idx])
            if citer not in test_papers and citee not in test_papers:
                val_cites.append([citer_idx, citee_idx])
                if citer not in val_papers and citee not in val_papers:
                    train_cites.append([citer_idx, citee_idx])
    train_cites = np.array(train_cites).T
    test_cites = np.array(test_cites).T
    val_cites = np.array(val_cites).T


    def word_to_idx(word):
        '''
        words all formatted like: "word1328"
        '''
        return int(word[4:]) - 1

    train_content = []
    test_content = []
    val_content = []
    with open(csv_file_str.format('content')) as contentfile:
        reader = csv.reader(contentfile)
        for paper_name, word_name in reader:
            paper_idx = paper_name_to_idx[paper_name]
            word_idx = word_to_idx(word_name)
            test_content.append([paper_idx, word_idx])
            if paper_name not in test_papers:
                val_content.append([paper_idx, word_idx])
                if paper_name not in val_papers:
                    train_content.append([paper_idx, word_idx])
    train_content = np.array(train_content).T
    test_content = np.array(test_content).T
    val_content = np.array(val_content).T

    n_classes = len(class_names)
    n_papers = len(paper_names)
    n_words = len(word_names)
    ent_papers = Entity(0, n_papers)
    #ent_classes = Entity(1, n_classes)
    ent_words = Entity(1, n_words)
    rel_paper = Relation(2, [ent_papers, ent_papers], is_set=True)
    rel_cites = Relation(0, [ent_papers, ent_papers])
    rel_content = Relation(1, [ent_papers, ent_words])
    schema = DataSchema([ent_papers, ent_words], [rel_cites, rel_content])
    schema_out = DataSchema([ent_papers], [rel_paper])

    train_vals = {'data_indices': train_indices, 'paper': train_paper, 'cites': train_cites, 'content': train_content}
    val_vals = {'data_indices': all_val_indices, 'paper': val_paper, 'cites': val_cites, 'content': val_content}
    test_vals = {'data_indices': all_test_indices, 'paper': test_paper, 'cites': test_cites, 'content': test_content}

    #%%
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, train_targets = get_data_and_targets(schema, args.neg_data, **train_vals)
    train_data = train_data.to(device)
    train_targets = train_targets.to(device)
    indices_identity, indices_transpose = train_data.calculate_indices()

    val_data, val_targets = get_data_and_targets(schema, args.neg_data, **val_vals)
    val_data = val_data.to(device)
    val_targets = val_targets.to(device)
    idx_id_val, idx_trans_val = val_data.calculate_indices()

    test_data, test_targets = get_data_and_targets(schema, args.neg_data, **test_vals)
    test_data = test_data.to(device)
    test_targets = test_targets.to(device)
    idx_id_test, idx_trans_test = test_data.calculate_indices()

    if args.training_data == 'val':
        train_data = val_data
        indices_identity = idx_id_val
        indices_transpose = idx_trans_val
    elif args.training_data == 'test':
        train_data = test_data
        indices_identity = idx_id_test
        indices_transpose = idx_trans_test

        val_data = test_data
        idx_id_val = indices_identity
        idx_trans_val = indices_transpose

    data_target = Data(schema_out)
    data_target[0] = SparseMatrix(indices = torch.arange(len(paper_names), dtype=torch.int64).repeat(2,1),
                                   values=torch.zeros([len(paper_names), n_classes]),
                                   shape=(len(paper_names), len(paper_names), n_classes))
    data_target = data_target.to(device)

    #%%

    # Loss function:
    def classification_loss(data_pred, data_true):
        return F.cross_entropy(data_pred, data_true)

    n_channels = 1
    net = SparseMatrixEntityPredictor(schema, n_channels,
                                         layers = args.layers,
                                         fc_layers=args.fc_layers,
                                         activation=eval('nn.%s()' % args.act_fn),
                                         final_activation = nn.Identity(),
                                         target_entities=schema_out.entities,
                                         dropout=args.dropout_rate,
                                         output_dim=n_classes,
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
    args.update({'dataset': 'cora'})
    if args.wandb_log_run:
        wandb.init(config=args,
            project="EquivariantRelational",
            entity='danieltlevy',
            settings=wandb.Settings(start_method='fork'))
        wandb.watch(net, log='all', log_freq=args.wandb_log_param_freq)

    PATH = "models/" + args.checkpoint_path
    val_acc_best = 0

    progress = tqdm(range(args.num_epochs), desc="Epoch 0", position=0, leave=True)

    for epoch in progress:
        net.train()
        opt.zero_grad()
        data_out = net(train_data, indices_identity, indices_transpose, data_target)
        data_out_values = data_out[train_indices]
        train_loss = classification_loss(data_out_values, train_targets)
        train_loss.backward()
        opt.step()
        with torch.no_grad():
            acc = (data_out_values.argmax(1) == train_targets).sum() / len(train_targets)
            progress.set_description(f"Epoch {epoch}")
            progress.set_postfix(loss=train_loss.item(), train_acc=acc.item())
            wandb_log = {'Train Loss': train_loss.item(), 'Train Accuracy': acc.item()}
            if epoch % args.val_every == 0:
                net.eval()
                data_out_val = net(val_data, idx_id_val, idx_trans_val, data_target)
                data_out_val_values = data_out_val[val_indices]
                val_loss = classification_loss(data_out_val_values, val_targets)
                val_acc = (data_out_val_values.argmax(1) == val_targets).sum() / len(val_targets)
                print("\nVal Acc: {:.3f} Val Loss: {:.3f}".format(val_acc, val_loss))
                wandb_log.update({'Val Loss': val_loss.item(), 'Val Accuracy': val_acc.item()})
                if not args.no_scheduler:
                    sched.step(val_loss)
                if val_acc > val_acc_best:
                    print("Saving")
                    val_acc_best = val_acc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'loss': train_loss.item(),
                        'val_acc_best': val_acc_best.item()
                        }, PATH)
            if epoch % args.wandb_log_loss_freq == 0:
                if args.wandb_log_run:
                    wandb.log(wandb_log)
