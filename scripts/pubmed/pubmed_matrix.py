# -*- coding: utf-8 -*-

'''
Source of Pubmed dataset, as well as dataset preparation procedures:
https://arxiv.org/pdf/2004.00216.pdf
https://github.com/yangji9181/HNE
May be downloaded via:
    ```
    pip install gdown
    gdown https://drive.google.com/uc?id=1ZEi2sTaZ2bk8cQwyCxtlwuWJsAq9N-Cl
    ```
'''
import sys
from scripts.pubmed.PubMedDataLoader import PubMedData
from src.SparseMatrix import SparseMatrix
from src.EquivariantNetwork import SparseMatrixEntityPredictor
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
import random
import pdb
import argparse
import wandb
import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterDict is not supported.")

def get_hyperparams(argv):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.set_defaults(dataset='PubMed')
    parser.add_argument('--checkpoint_path', type=str, default='pubmed_node.pt')
    parser.add_argument('--layers', type=int, nargs='*', default=['64']*3,
                        help='Number of channels for equivariant layers')
    parser.add_argument('--fc_layers', type=str, nargs='*', default=[50],
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
    parser.set_defaults(norm_affine=True)
    parser.add_argument('--pool_op', type=str, default='mean')
    parser.add_argument('--neg_data', type=float, default=0.,
                        help='Ratio of random data samples to positive. \
                              When sparse, this is similar to number of negative samples')
    parser.add_argument('--training_data', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--val_pct', type=float, default=10.)
    parser.add_argument('--test_pct', type=float, default=0.)
    parser.add_argument('--semi_supervised', action='store_true', help='switch to low-label regime')
    parser.set_defaults(semi_supervised=False)
    parser.add_argument('--node_labels', dest='node_labels', action='store_true')
    parser.add_argument('--no_node_labels', dest='node_labels', action='store_false')
    parser.set_defaults(node_labels=True)
    parser.add_argument('--wandb_log_param_freq', type=int, default=250)
    parser.add_argument('--wandb_log_loss_freq', type=int, default=1)
    parser.add_argument('--wandb_log_run', dest='wandb_log_run', action='store_true',
                        help='Log this run in wandb')
    parser.add_argument('--wandb_no_log_run', dest='wandb_log_run', action='store_false',
                        help='Do not log this run in wandb')
    parser.add_argument('--output', type=str)
    parser.set_defaults(wandb_log_run=False)

    args, argv = parser.parse_known_args(argv)
    if args.output == None:
        args.output = args.dataset + '_emb.dat'
    args.layers  = [int(x) for x in args.layers]
    args.fc_layers = [int(x) for x in args.fc_layers]
    return args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
#%%
def set_seed(seed):
    random.seed(seed, version=2)
    np.random.seed(random.randint(0, 2**32))
    torch.manual_seed(random.randint(0, 2**32))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_embeddings(args, embeddings, node_idx_to_id):
    print("Saving embeddings")
    with open(args.output, 'w') as file:
        file.write(str(args) + '\n')
        for idx in range(embeddings.shape[0]):
            embedding = embeddings[idx]
            name = node_idx_to_id[idx]
            file.write('{}\t{}\n'.format(name, ' '.join(embedding.astype(str))))


if __name__ == '__main__':
    argv = sys.argv[1:]
    args = get_hyperparams(argv)
    print(args)
    set_seed(args.seed)

    dataloader = PubMedData(args.node_labels)
    schema = dataloader.schema
    data = dataloader.data.to(device)
    data_target = dataloader.data_target.to(device)
    indices_identity, indices_transpose = data.calculate_indices()
    targets = dataloader.targets.to(device)
    schema_out = dataloader.schema_out
    n_targets = targets.shape[0]
    target_indices = dataloader.target_indices
    n_outputs = dataloader.n_outputs
    targets = dataloader.targets
    target_node_idx_to_id = dataloader.target_node_idx_to_id

    #%%
    input_channels = {rel.id: data[rel.id].n_channels for rel in schema.relations}

    shuffled_indices_idx = random.sample(range(n_targets), n_targets)
    val_start = 0
    train_start = int(args.val_pct * (n_targets/100.))

    val_indices_idx  = shuffled_indices_idx[val_start:train_start]
    val_indices = target_indices[val_indices_idx]

    train_indices_idx = shuffled_indices_idx[train_start:]
    train_indices = target_indices[train_indices_idx]

    #%%
    train_targets = targets[train_indices_idx]
    val_targets = targets[val_indices_idx]


    n_output_classes = len(targets.unique())
    data_target[0] = SparseMatrix(
            indices = torch.arange(n_outputs, dtype=torch.int64).repeat(2,1),
            values=torch.zeros([n_outputs, n_output_classes]),
            shape=(n_outputs, n_outputs, n_output_classes),
            is_set=True)


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

    def f1_scores(values, target):
        micro = f1_score(values.argmax(1).cpu(), target.cpu(), average='micro')
        macro = f1_score(values.argmax(1).cpu(), target.cpu(), average='macro')
        return micro, macro

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
                val_micro, val_macro = f1_scores(data_out_val_values, val_targets)
                print("\nVal Acc: {:.3f} Val Loss: {:.3f} Val Micro-F1: {:.3f} \
 Val Macro-F1: {:.3f}".format(val_acc, val_loss, val_micro, val_macro))
                wandb_log.update({'Val Loss': val_loss.item(), 'Val Accuracy': val_acc,
                                  'Val Micro-F1': val_micro, 'Val Macro-F1': val_macro})
                if val_acc > val_acc_best:
                    val_acc_best = val_acc
                    print("New best, saving")
                    torch.save({
                        'epoch': epoch,
                        'net_state_dict': net.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'train_loss': train_loss.item(),
                        'train_acc': acc,
                        'val_loss': val_loss.item(),
                        'val_acc': val_acc
                        }, args.checkpoint_path)
                    if args.wandb_log_run:
                        wandb.save(args.checkpoint_path)
                if not args.no_scheduler:
                    sched.step(val_loss)
            if epoch % args.wandb_log_loss_freq == 0:
                if args.wandb_log_run:
                    wandb.log(wandb_log)
    checkpoint = torch.load(args.checkpoint_path)
    net.load_state_dict(checkpoint['net_state_dict'])
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    net.eval()
    target_embeddings = net(data, indices_identity, indices_transpose,
                            data_target, get_embeddings=True).squeeze().detach().cpu().numpy()
    save_embeddings(args, target_embeddings, target_node_idx_to_id)
