#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import numpy as np
import torch
from src.GenerateData import RandomSparseData
from src.SparseTensor import SparseTensor
from src.SparseEquivariantLayer import SparseEquivariantLayerBlock, SparseEquivariantLayer
from src.EquivariantNetwork import SparseEquivariantNetwork
import torch.optim as optim
from tqdm import tqdm
import pdb
import logging
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

#%%
N_STUDENT = 50
N_COURSE = 30
N_PROFESSOR = 25
EMBEDDING_DIMS = 2
BATCH_SIZE = 1
SPARSITY = 0.01
CHANNELS_IN = 2
CHANNELS_OUT = 3

data_generator = RandomSparseData((N_STUDENT, N_COURSE, N_PROFESSOR),
                               sparsity=SPARSITY, n_channels=CHANNELS_IN)
sparse_data = data_generator.observed
schema = sparse_data.schema
relations = schema.relations


# The relations:
#Relation(0, [ent_students, ent_courses], 1)
#Relation(1, [ent_students, ent_professors], 1)
#Relation(2, [ent_professors, ent_courses], 1)
#Relation(3, [ent_students, ent_professors, ent_courses], 1)
#Relation(4, [ent_courses, ent_courses], 1)
#Relation(5, [ent_students], 1)
#Relation(6, [ent_students, ent_students, ent_students, ent_courses], 1)

#%% Test individual relation
logging.debug("test")
'''
relation_in_idx = 6
relation_out_idx = 6

sparse_in = sparse_data[relation_in_idx]
sparse_out = sparse_data[relation_out_idx]
layer =  SparseEquivariantLayerBlock(CHANNELS_IN, CHANNELS_OUT, schema,
                                    relations[relation_in_idx], relations[relation_out_idx])
layer(sparse_in, sparse_out)
'''

#%% Test whole layer
#layer1 = SparseEquivariantLayer(schema, input_dim=CHANNELS_IN, output_dim=CHANNELS_OUT)
#layer2 = SparseEquivariantLayer(schema, input_dim=CHANNELS_OUT, output_dim=CHANNELS_IN)
#out1 = layer1(sparse_data)





#%%
net = SparseEquivariantNetwork(schema, CHANNELS_IN)

# Loss functions:
def loss_fcn(data_pred, data_true):
    loss = torch.zeros(1).to(device)
    for relation in relations:
        rel_id = relation.id
        if data_pred[rel_id].nnz() == 0 or data_true[rel_id].nnz() == 0:
            continue
        else:
            loss += torch.mean((data_pred[rel_id].values - data_true[rel_id].values)**2)
    loss = loss / len(relations)
    return loss


learning_rate = 1e-3
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                             factor=0.5,
                                             patience=10,
                                             verbose=True)

#%%
epochs=1000
progress = tqdm(range(epochs), desc="Loss: ", position=0, leave=True)
for i in progress:
    optimizer.zero_grad()
    data_out = net.forward(sparse_data)
    train_loss = loss_fcn(data_out, sparse_data)
    train_loss.backward()
    optimizer.step()
    progress.set_description("Train: {:.4f}".format(train_loss.item()))
