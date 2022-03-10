#%%
import sys
sys.path.append('../../')
sys.path.append('../')

import time
import argparse
import wandb
from tqdm import tqdm
import pdb
from sklearn.metrics import f1_score
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from utils import EarlyStopping
from EquivHGNet import EquivHGNetAblation
from src.SparseMatrix import SparseMatrix
from src.utils import count_parameters, get_hyperparams

from data_nc import load_data, load_data_flat
import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterDict is not supported.")


def set_seed(seed):
    random.seed(seed, version=2)
    np.random.seed(random.randint(0, 2**32))
    torch.manual_seed(random.randint(0, 2**32))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_features(data, schema, feats_type, target_ent):
    '''
    TODO: IMPLEMENT THIS
    '''
    # Select features for nodes
    in_dims = {}
    ent_start = max(schema.relations.keys()) - len(schema.entities) + 1

    if feats_type == 0:
        # Keep all node attributes
        pass
    elif feats_type == 1:
        # Set all non-target node attributes to zero
        for ent_i in schema.entities:
            if ent_i.id != target_ent:
                # 10 dimensions for some reason
                n_dim = 10
                rel_id = ent_start + ent_i.id
                data[rel_id] = SparseMatrix.from_other_sparse_matrix(data[rel_id], n_dim)

    '''
    elif feats_type == 2:
        # Set all non-target node attributes to one-hot vector
        for i in range(0, len(features_list)):
            if i != target_ent:
                dim = features_list[i].shape[0]
                indices = torch.arange(n_instances).unsqueeze(0).repeat(2, 1)
                values = torch.FloatTensor(np.ones(dim))
                features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = np.ones(dim)
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    '''
    for rel_id in schema.relations:
        in_dims[rel_id] = data[rel_id].n_channels
    return data, in_dims

def remove_extra_relations(dataset, schema, data, flat=False):
    if dataset == 'ACM':
        redundant = [1, 3, 5, 7]
    elif dataset == 'DBLP':
        redundant = [3, 4, 5]
    elif dataset == 'IMDB':
        redundant = [1, 3, 5]
    elif dataset == 'Freebase':
        redundant = []
    if flat:
        for rel in sorted(redundant, reverse=True):
            data[0].values = torch.cat([data[0].values[:, :rel],
                                        data[0].values[:, rel+1:]], dim=1)
        data[0].n_channels -= len(redundant)
    else:
        for rel in sorted(redundant, reverse=True):
            del(schema.relations[rel])
            del(data[rel])

    return schema, data

def regr_fcn(logits, multi_label=False):
    if multi_label:
        return torch.sigmoid(logits)
    else:
        return F.log_softmax(logits, 1)

def loss_fcn(data_pred, data_true, multi_label=False):
    if multi_label:
        return F.binary_cross_entropy(data_pred, data_true)
    else:
        return F.nll_loss(data_pred, data_true)


def f1_scores(logits, target):
    values = logits.argmax(1).detach().cpu()
    micro = f1_score(target.cpu(), values, average='micro')
    macro = f1_score(target.cpu(), values, average='macro')
    return micro, macro

def f1_scores_multi(logits, target):
    values = (logits.detach().cpu().numpy()>0).astype(int)
    micro = f1_score(target, values, average='micro')
    macro = f1_score(target, values, average='macro')
    return micro, macro

def pred_fcn(values, multi_label=False):
    if multi_label:
        pass
    else:        
        values.cpu().numpy().argmax(axis=1)
#%%
def run_model(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.lgnn:
        load_data_fn = load_data_flat
    else:
        load_data_fn = load_data
    schema, schema_out, data, data_target, labels, \
        train_val_test_idx, dl = load_data_fn(args.dataset,
                       use_edge_data=args.use_edge_data,
                       use_node_attrs=args.use_node_attr,
                       feats_type=args.feats_type)
    if args.asymmetric:
        schema, data = remove_extra_relations(args.dataset, schema, data, args.lgnn)
    target_entity_id = 0 # True for all current NC datasets
    target_entity = schema.entities[target_entity_id]
    data, in_dims = select_features(data, schema, args.feats_type, target_entity_id)
    if args.multi_label:
        labels = torch.FloatTensor(labels).to(device)
    else:
        labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)
    
    data = data.to(device)
    indices_identity, indices_transpose = data.calculate_indices()
    data_target = data_target.to(device)

    num_classes = dl.labels_train['num_classes']
    net = EquivHGNetAblation(schema, in_dims,
                        layers = args.layers,
                        in_fc_layer=args.in_fc_layer,
                        fc_layers=args.fc_layers,
                        activation=eval('nn.%s()' % args.act_fn),
                        final_activation = nn.Identity(),
                        target_entities=[target_entity],
                        dropout=args.dropout,
                        output_dim=num_classes,
                        norm=args.norm,
                        pool_op=args.pool_op,
                        norm_affine=args.norm_affine,
                        norm_out=args.norm_out,
                        residual=args.residual,
                        mid_fc_layer=args.mid_fc_layer,
                        removed_params = args.removed_params)

    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)


    if args.wandb_log_run:
        wandb.init(config=args,
            settings=wandb.Settings(start_method='fork'),
            project="EquivariantHGN",
            entity='danieltlevy')
        wandb.watch(net, log='all', log_freq=args.wandb_log_param_freq)
    print(args)
    print("Number of parameters: {}".format(count_parameters(net)))

    run_name = args.dataset + '_' + str(args.run)
    if args.wandb_log_run and wandb.run.name is not None:
        run_name = run_name + '_' + str(wandb.run.name)

    if args.checkpoint_path != '':
        checkpoint_path = args.checkpoint_path
    else:
        checkpoint_path = f"checkpoint/checkpoint_{run_name}.pt"

    print("Checkpoint Path: " + checkpoint_path)
    progress = tqdm(range(args.epoch), desc="Epoch 0", position=0, leave=True)
    # training loop
    net.train()
    val_micro_best = 0
    for epoch in progress:
        # training
        net.train()
        optimizer.zero_grad()
        logits = net(data, indices_identity, indices_transpose,
                     data_target).squeeze()
        logp = regr_fcn(logits, args.multi_label)
        train_loss = loss_fcn(logp[train_idx], labels[train_idx], args.multi_label)
        train_loss.backward()
        optimizer.step()
        if args.multi_label:
            train_micro, train_macro = f1_scores_multi(logits[train_idx],
                                                 dl.labels_train['data'][train_idx])
        else:
            train_micro, train_macro = f1_scores(logits[train_idx],
                                                 labels[train_idx])
        with torch.no_grad():
            progress.set_description(f"Epoch {epoch}")
            progress.set_postfix(loss=train_loss.item(), micr=train_micro)
            wandb_log = {'Train Loss': train_loss.item(),
                         'Train Micro': train_micro,
                         'Train Macro': train_macro}
            if epoch % args.val_every == 0:
                # validation
                net.eval()
                logits = net(data, indices_identity, indices_transpose, data_target).squeeze()
                logp = regr_fcn(logits, args.multi_label)
                val_loss = loss_fcn(logp[val_idx], labels[val_idx], args.multi_label)
                if args.multi_label:
                    val_micro, val_macro = f1_scores_multi(logits[val_idx],
                                                         dl.labels_train['data'][val_idx])
                else:
                    val_micro, val_macro = f1_scores(logits[val_idx],
                                                         labels[val_idx])
                print("\nVal Loss: {:.3f} Val Micro-F1: {:.3f} \
Val Macro-F1: {:.3f}".format(val_loss, val_micro, val_macro))
                wandb_log.update({'Val Loss': val_loss.item(),
                                  'Val Micro-F1': val_micro, 'Val Macro-F1': val_macro})
                if val_micro > val_micro_best:

                    val_micro_best = val_micro
                    print("New best, saving")
                    torch.save({
                        'epoch': epoch,
                        'net_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss.item(),
                        'train_micro': train_micro,
                        'train_macro': train_macro,
                        'val_loss': val_loss.item(),
                        'val_micro': val_micro,
                        'val_macro': val_macro
                        }, checkpoint_path)
                    if args.wandb_log_run:
                        wandb.summary["val_micro_best"] = val_micro
                        wandb.summary["val_macro_best"] = val_macro
                        wandb.summary["val_loss_best"] = val_loss.item()
                        wandb.summary["epoch_best"] = epoch
                        wandb.summary["train_loss_best"] = train_loss.item()
                        wandb.summary['train_micro_best'] = train_micro
                        wandb.summary['train_macro_best'] = train_macro
                        wandb.save(checkpoint_path)

            if epoch % args.wandb_log_loss_freq == 0:
                if args.wandb_log_run:
                    wandb.log(wandb_log, step=epoch)


    # testing with evaluate_results_nc
    if args.evaluate:

        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net_state_dict'])
        net.eval()
        test_logits = []
        with torch.no_grad():
            logits = net(data, indices_identity, indices_transpose,
                         data_target).squeeze()
            test_logits = logits[test_idx]
            if args.multi_label:
                pred = (test_logits.cpu().numpy()>0).astype(int)
            else:
                pred = test_logits.cpu().numpy().argmax(axis=1)
                onehot = np.eye(num_classes, dtype=np.int32)

            file_path = f"test_out/{run_name}.txt"
            dl.gen_file_for_evaluate(test_idx=test_idx, label=pred,
                                     file_path=file_path,
                                     multi_label=args.multi_label)
            if not args.multi_label:
                pred = onehot[pred]
            print(dl.evaluate(pred))

#%%
if __name__ == '__main__':
    argv = sys.argv[1:]
    args = get_hyperparams(argv)
    set_seed(args.seed)
    #%%
    run_model(args)
