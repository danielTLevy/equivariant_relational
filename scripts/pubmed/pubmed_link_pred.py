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
from scripts.pubmed.PubMedDataLoader import PubMedData, TARGET_NODE_TYPE, TARGET_REL_ID
from src.DataSchema import DataSchema, Entity, Relation, SparseMatrixData
from src.SparseMatrix import SparseMatrix
from src.EquivariantNetwork import SparseMatrixAutoEncoder
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
    parser.add_argument('--embedding_dim', type=int, default=50)
    parser.add_argument('--neg_data', type=float, default=0.,
                        help='Ratio of random data samples to positive. \
                              When sparse, this is similar to number of negative samples')
    parser.add_argument('--training_data', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--val_pct', type=float, default=10.)
    parser.add_argument('--test_pct', type=float, default=0.)
    parser.add_argument('--target_n_samples', type=int, default=1000)
    parser.add_argument('--target_pos_rate', type=float, default=0.5)

    parser.add_argument('--semi_supervised', action='store_true', help='switch to low-label regime')
    parser.set_defaults(semi_supervised=False)
    parser.add_argument('--node_labels', dest='node_labels', action='store_true')
    parser.add_argument('--no_node_labels', dest='node_labels', action='store_false')
    parser.set_defaults(node_labels=True)
    parser.add_argument('--save_embeddings', dest='save_embeddings', action='store_true', default=True)
    parser.add_argument('--no_save_embeddings', dest='save_embeddings', action='store_false', default=True)
    parser.set_defaults(save_embeddings=True)
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

def generate_target_matrix(true_matrix, n_samples, pos_rate, device):
    '''
    Generate a target matrix with n_samples indices, of which pos_rate
    is the proportion are true positives, while 1-pos_rate is the proportion
    of randomly generated links.
    true_matrix is a matrix containing all true positive links
    Note that the randomly generated links have a ~99.99% of being negative but
    there may be some false negatives (1e-4 sparsity for each relation)
    '''
    n_n = true_matrix.n
    n_m = true_matrix.m
    n_channels = 1
    n_pos_samples = int(pos_rate * n_samples)
    perm = torch.randperm(true_matrix.nnz())
    pos_sample_idx = perm[:n_pos_samples]
    pos_indices = true_matrix.indices[:, pos_sample_idx]
    pos_values = torch.ones(n_pos_samples).to(device)
    
    n_neg_samples = n_samples - n_pos_samples
    neg_indices_n = torch.randint(0, n_n, [n_neg_samples]).to(device)
    neg_indices_m = torch.randint(0, n_m, [n_neg_samples]).to(device)
    neg_indices = torch.stack((neg_indices_n, neg_indices_m))
    neg_values = torch.zeros(n_neg_samples).to(device)
    
    return SparseMatrix(
            indices = torch.cat((pos_indices, neg_indices), 1),
            values = torch.cat((pos_values, neg_values), 0).unsqueeze(1),
            shape = (n_n, n_m, n_channels)).coalesce()

if __name__ == '__main__':
    argv = sys.argv[1:]
    args = get_hyperparams(argv)
    print(args)
    set_seed(args.seed)

    dataloader = PubMedData(args.node_labels)
    schema = dataloader.schema
    data = dataloader.data.to(device)
    indices_identity, indices_transpose = data.calculate_indices()
    embedding_entity = schema.entities[TARGET_NODE_TYPE]
    input_channels = {rel.id: data[rel.id].n_channels for rel in schema.relations}
    embedding_schema = DataSchema(schema.entities,
                                  Relation(0,
                                           [embedding_entity, embedding_entity],
                                           is_set=True))
    n_instances = embedding_entity.n_instances
    data_embedding = SparseMatrixData(embedding_schema)
    data_embedding[0] = SparseMatrix(
            indices = torch.arange(n_instances, dtype=torch.int64).repeat(2,1),
            values=torch.zeros([n_instances, args.embedding_dim]),
            shape=(n_instances, n_instances, args.embedding_dim),
            is_set=True)
    data_embedding.to(device)
    target_schema = DataSchema(schema.entities, schema.relations[TARGET_REL_ID])
    target_node_idx_to_id = dataloader.target_node_idx_to_id
    #%%
    net = SparseMatrixAutoEncoder(schema, input_channels,
                                         layers = args.layers,
                                         embedding_dim=args.embedding_dim,
                                         embedding_entities=[embedding_entity],
                                         activation=eval('nn.%s()' % args.act_fn),
                                         final_activation = nn.Sigmoid(),
                                         dropout=args.dropout_rate,
                                         norm=args.norm,
                                         pool_op=args.pool_op,
                                         norm_affine=args.norm_affine,
                                         output_relations=[target_schema.relations])
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

    def reconstruction_loss(data_pred, data_true):
        return torch.sum((data_pred - data_true)**2)

    def acc_fcn(data_pred, data_true):
        # TODO: return AUC ROC
        return (((data_pred > 0.5) == data_true).sum() / len(data_true)).item()

    def generate_target():
        target_matrix = generate_target_matrix(data[TARGET_REL_ID],
                                               args.target_n_samples,
                                               args.target_pos_rate,
                                               device)
        data_target = SparseMatrixData(target_schema)
        data_target[TARGET_REL_ID] = target_matrix
        data_target.to(device)
        return data_target

    for epoch in progress:
        net.train()
        opt.zero_grad()
        data_target = generate_target()
        data_out = net(data, indices_identity, indices_transpose,
                       data_target, data_embedding)
        data_out_values = data_out[TARGET_REL_ID].values.squeeze()
        data_target_values = data_target[TARGET_REL_ID].values.squeeze()
        loss = reconstruction_loss(data_out_values, data_target_values)
        loss.backward()
        opt.step()
        with torch.no_grad():
            acc = acc_fcn(data_out_values, data_target_values)
            progress.set_description(f"Epoch {epoch}")
            progress.set_postfix(loss=loss.item(), acc=acc)
            loss_val = loss.cpu().item()
            wandb_log = {'Loss': loss_val, 'Accuracy': acc}
            if epoch % args.wandb_log_loss_freq == 0:
                if args.wandb_log_run:
                    wandb.log(wandb_log)

    if args.save_embeddings:
        net.eval()
        target_embeddings = net(data, indices_identity, indices_transpose,
                                data_target, get_embeddings=True).squeeze().detach().cpu().numpy()
        save_embeddings(args, target_embeddings, target_node_idx_to_id)
