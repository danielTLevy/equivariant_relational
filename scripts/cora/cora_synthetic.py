#%%
import sys
from src.DataSchema import DataSchema, Entity, Relation, SparseMatrixData, Data
from src.SparseMatrix import SparseMatrix
from src.EquivariantNetwork import SparseMatrixEquivariantNetwork
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import argparse
import random
import pdb
import time

#%%
csv_file_str = './data/cora/{}.csv'

def get_hyperparams(argv):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--checkpoint_path', type=str, required=False)
    parser.add_argument('--layers', type=int, nargs='*', default=['16']*3,
                        help='Number of channels for equivariant layers')
    parser.add_argument('--fc_layers', type=str, nargs='*', default=[],
                        help='Fully connected layers for target embeddings')
    parser.add_argument('--l2_decay', type=float, default=0)
    parser.add_argument('--dropout_rate', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--act_fn', type=str, default='ReLU')
    parser.add_argument('--sched_factor', type=float, default=0.5)
    parser.add_argument('--sched_patience', type=float, default=300)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--val_every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--norm',  dest='norm', action='store_true', default=True)
    parser.add_argument('--no_norm', dest='norm', action='store_false', default=True)
    parser.set_defaults(norm=True)
    parser.add_argument('--neg_data', type=float, default=1.,
                        help='Ratio of random data samples to positive. \
                              When sparse, this is similar to number of negative samples'),
    parser.add_argument('--pool_op', type=str, default='add')
    
    args, argv = parser.parse_known_args(argv)
    args.layers  = [int(x) for x in args.layers]
    args.fc_layers = [int(x) for x in args.fc_layers]

    return args


def set_seed(seed):
    random.seed(seed, version=2)
    np.random.seed(random.randint(0, 2**32))
    torch.manual_seed(random.randint(0, 2**32))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#%%
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    argv = sys.argv[1:]
    args = get_hyperparams(argv)

    print(args)
    set_seed(args.seed)

    n_papers = 200
    n_words = 300
    n_classes = 4
    
    value_dist = torch.distributions.bernoulli.Bernoulli(probs=1./(1.+args.neg_data))
    paper = np.stack([np.arange(n_papers),
                      np.random.randint(0, n_classes, n_papers)])

    n_cites = 2*int(n_papers*3)
    cites = np.unique(np.random.randint(0, n_papers, (2, n_cites)), axis=1)
    cites_matrix = SparseMatrix(
            indices = torch.LongTensor(cites),
            values = value_dist.sample((cites.shape[1], 1)),
            shape = (n_papers, n_papers, 1)
            ).coalesce()
    n_content = 2*int(0.2*n_papers*n_words)
    
    content = np.stack([np.random.randint(0, n_papers, (n_content)),
                      np.random.randint(0, n_words, (n_content))])
    content = np.unique(cites, axis=1)
    content_matrix = SparseMatrix(
              indices = torch.LongTensor(content),
              values = value_dist.sample((content.shape[1], 1)),
              shape = (n_papers, n_words, 1)
              ).coalesce()

    ent_papers = Entity(0, n_papers)
    #ent_classes = Entity(1, n_classes)
    ent_words = Entity(1, n_words)
    rel_paper = Relation(0, [ent_papers, ent_papers], is_set=True)
    rel_cites = Relation(0, [ent_papers, ent_papers])
    rel_content = Relation(1, [ent_papers, ent_words])
    schema = DataSchema([ent_papers, ent_words], [rel_cites, rel_content])
    schema_out = DataSchema([ent_papers], [rel_paper])
    targets = torch.LongTensor(paper[1])

    data = SparseMatrixData(schema)
    data[0] = cites_matrix
    data[1] = content_matrix
    

    indices_identity, indices_transpose = data.calculate_indices()

    data_target = Data(schema_out)
    data_target[0] = SparseMatrix(indices = torch.arange(n_papers, dtype=torch.int64).repeat(2,1),
                                   values=torch.zeros([n_papers, n_classes]),
                                   shape=(n_papers, n_papers, n_classes))
    data_target = data_target.to(device)

    #%%

    # Loss function:
    def classification_loss(data_pred, data_true):
        return F.cross_entropy(data_pred, data_true)

    n_channels = 1
    net = SparseMatrixEquivariantNetwork(schema, n_channels,
                                         layers = args.layers,
                                         fc_layers=args.fc_layers,
                                         activation=eval('nn.%s()' % args.act_fn),
                                         final_activation = nn.Identity(),
                                         target_entities=schema_out.entities,
                                         dropout=args.dropout_rate,
                                         output_dim=n_classes,
                                         norm=args.norm)
    net = net.to(device)

    opt = eval('optim.%s' % args.optimizer)(net.parameters(), lr=args.learning_rate, weight_decay=args.l2_decay)

    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                 factor=args.sched_factor,
                                                 patience=args.sched_patience,
                                                 verbose=True)

    #%%
    n_epochs = 3000
    progress = tqdm(range(n_epochs), desc="Epoch 0", position=0, leave=True)
    net.train()

    for epoch in progress:
        #time.sleep(1)
        opt.zero_grad()
        data_out = net(data, indices_identity, indices_transpose, data_target)
        data_out_values = data_out
        train_loss = classification_loss(data_out_values, targets)
        train_loss.backward()
        opt.step()
        acc = (data_out_values.argmax(1) == targets).sum() / len(targets)
        progress.set_description(f"Epoch {epoch}")
        progress.set_postfix(loss=train_loss.item(), train_acc=acc.item())
        sched.step(train_loss.item())

