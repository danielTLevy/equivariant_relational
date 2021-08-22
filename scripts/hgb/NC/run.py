import sys
sys.path.append('../../')
sys.path.append('../')

import time
import argparse
import wandb
from tqdm import tqdm
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import EarlyStopping
from EquivHGNet import EquivHGNet
#from utils.pytorchtools import EarlyStopping
from data import load_data
import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterDict is not supported.")

def select_features(data, schema, feats_type, target_ent):
    '''
    TODO: IMPLEMENT THIS
    '''
    # Select features for nodes
    entities = schema
    features_list = data
    in_dims = {}
    num_relations = len(schema.relations) - len(schema.entities)
    '''
    if feats_type == 1:
        in_dims = []#[features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i != target_ent:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)

    elif feats_type == 2:
        in_dims = [features.shape[0] for features in features_list]
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
    for rel in schema.relations:
        in_dims[rel.id] = data[rel.id].n_channels
    return data, in_dims

def loss_fcn(data_pred, data_true):
    return F.nll_loss(data_pred, data_true)

def acc_fcn(values, target):
    return ((values.argmax(1) == target).sum() / len(target)).item()
    
    
#%%
def run_model_DBLP(args):
    feats_type = args.feats_type
    loaded = load_data(args.dataset)
    schema, schema_out, data, data_target, labels, train_val_test_idx, dl = loaded
    target_entity_id = 0 # TODO: figure out if this is true for all datasets
    target_entity = schema.entities[target_entity_id]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data, in_dims = select_features(data, schema, args.feats_type, target_entity_id)
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

    for _ in range(args.repeat):
        num_classes = dl.labels_train['num_classes']
        net = EquivHGNet(schema, in_dims,
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
                            norm_affine=args.norm_affine)

        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)


        if args.wandb_log_run:
            wandb.init(config=args,
                project="EquivariantHGN",
                entity='danieltlevy')
            wandb.watch(net, log='all', log_freq=args.wandb_log_param_freq)
            
        progress = tqdm(range(args.epoch), desc="Epoch 0", position=0, leave=True)
        # training loop
        net.train()
        #early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path='checkpoint/checkpoint.pt')#'_{}_{}.pt'.format(args.dataset, args.num_layers))
        val_acc_best = 0
        for epoch in progress:
            # training
            net.train()
            optimizer.zero_grad()
            logits = net(data, indices_identity, indices_transpose,
                         data_target).squeeze()
            logp = torch.sigmoid(logits)
            train_loss = F.nll_loss(logp[train_idx], labels[train_idx])
            train_loss.backward()
            optimizer.step()
            with torch.no_grad():
                acc = acc_fcn(logp[train_idx], labels[train_idx])
                progress.set_description(f"Epoch {epoch}")
                progress.set_postfix(loss=train_loss.item(), train_acc=acc)
                wandb_log = {'Train Loss': train_loss.item(), 'Train Accuracy': acc}
                if epoch % args.val_every == 0:
                    # validation
                    net.eval()
                    logits = net(data, indices_identity, indices_transpose, data_target).squeeze()
                    logp = torch.sigmoid(logits)
                    val_loss = loss_fcn(logp[val_idx], labels[val_idx])
                    val_acc = acc_fcn(logp[val_idx], labels[val_idx])
                    print("\nVal Acc: {:.3f} Val Loss: {:.3f}".format(val_acc, val_loss))
                    wandb_log.update({'Val Loss': val_loss.item(), 'Val Accuracy': val_acc})
                    if val_acc > val_acc_best:
                        val_acc_best = val_acc
                        print("New best, saving")
                        torch.save({
                            'epoch': epoch,
                            'net_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': train_loss.item(),
                            'train_acc': acc,
                            'val_loss': val_loss.item(),
                            'val_acc': val_acc
                            }, args.checkpoint_path)
                        if args.wandb_log_run:
                            wandb.save(args.checkpoint_path)

                if epoch % args.wandb_log_loss_freq == 0:
                    if args.wandb_log_run:
                        wandb.log(wandb_log)


        # testing with evaluate_results_nc
        net.load_state_dict(torch.load('checkpoint/checkpoint.pt'))#_{}_{}.pt'.format(args.dataset, args.num_layers)))
        net.eval()
        test_logits = []
        with torch.no_grad():
            logits = net(data, indices_identity, indices_transpose,
                         data_target).squeeze()
            test_logits = logits[test_idx]
            pred = test_logits.cpu().numpy().argmax(axis=1)
            onehot = np.eye(num_classes, dtype=np.int32)
            dl.gen_file_for_evaluate(test_idx=test_idx, label=pred, file_name=f"{args.dataset}_{args.run}.txt")
            pred = onehot[pred]
            print(dl.evaluate(pred))
#%%
def get_hyperparams(argv):
    ap = argparse.ArgumentParser(allow_abbrev=False, description='EquivHGN for Node Classification')
    ap.set_defaults(dataset='PubMed')
    ap.add_argument('--feats-type', type=int, default=3,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others);' + 
                        '5 - only term features (zero vec for others).')
    ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=30, help='Patience.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--dataset', type=str, default='DBLP')
    ap.add_argument('--checkpoint_path', type=str, default='pubmed_node.pt')
    ap.add_argument('--layers', type=int, nargs='*', default=['64']*3,
                        help='Number of channels for equivariant layers')
    ap.add_argument('--fc_layers', type=str, nargs='*', default=[50],
                        help='Fully connected layers for target embeddings')
    ap.add_argument('--weight_decay', type=float, default=0.1)
    ap.add_argument('--act_fn', type=str, default='ReLU')
    ap.add_argument('--in_fc_layer',  dest='in_fc_layer', action='store_true', default=True)
    ap.add_argument('--no_in_fc_layer', dest='in_fc_layer', action='store_false', default=True)
    ap.set_defaults(in_fc_layer=True)
    ap.add_argument('--optimizer', type=str, default='Adam')
    ap.add_argument('--val_every', type=int, default=10)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--norm',  dest='norm', action='store_true', default=True)
    ap.add_argument('--no_norm', dest='norm', action='store_false', default=True)
    ap.set_defaults(norm=True)
    ap.add_argument('--norm_affine', action='store_true')
    ap.set_defaults(norm_affine=True)
    ap.add_argument('--pool_op', type=str, default='mean')
    ap.add_argument('--node_labels', dest='node_labels', action='store_true')
    ap.add_argument('--no_node_labels', dest='node_labels', action='store_false')
    ap.set_defaults(node_labels=True)
    ap.add_argument('--save_embeddings', dest='save_embeddings', action='store_true', default=True)
    ap.add_argument('--no_save_embeddings', dest='save_embeddings', action='store_false', default=True)
    ap.set_defaults(save_embeddings=True)
    ap.add_argument('--wandb_log_param_freq', type=int, default=250)
    ap.add_argument('--wandb_log_loss_freq', type=int, default=1)
    ap.add_argument('--wandb_log_run', dest='wandb_log_run', action='store_true',
                        help='Log this run in wandb')
    ap.add_argument('--wandb_no_log_run', dest='wandb_log_run', action='store_false',
                        help='Do not log this run in wandb')
    ap.add_argument('--output', type=str)
    ap.set_defaults(wandb_log_run=True)

    args, argv = ap.parse_known_args(argv)
    if args.output == None:
        args.output = args.dataset + '_emb.dat'
    args.layers  = [int(x) for x in args.layers]
    args.fc_layers = [int(x) for x in args.fc_layers]
    return args


#%%
if __name__ == '__main__':
    argv = sys.argv[1:]
    args = get_hyperparams(argv)
    run_model_DBLP(args)
