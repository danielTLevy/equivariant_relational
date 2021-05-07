#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
from src.GenerateData import SyntheticData
from src.SparseMatrix import SparseMatrix
from src.SparseMatrixEquivariantLayer import SparseMatrixEquivariantLayerBlock, SparseMatrixEquivariantLayer
from src.EquivariantNetwork import SparseMatrixEquivariantNetwork
import torch.optim as optim
from tqdm import tqdm
import logging

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

N_STUDENT = 200
N_COURSE = 300
N_PROFESSOR = 400
EMBEDDING_DIMS = 2

data_generator = SyntheticData((N_STUDENT, N_COURSE, N_PROFESSOR),
                               sparsity=0.9,  embedding_dims=EMBEDDING_DIMS, batch_dim=False).to(device)
data = data_generator.data
observed = data_generator.observed
missing = {key:  ~val for key, val in observed.items()}
schema = data.schema
relations = schema.relations

#%%
# hide unobserved
data_hidden = data.mask_data(observed).to_sparse_matrix().to(device)

#%%


    
#%%
net = SparseMatrixEquivariantNetwork(schema, 1).to(device)
#%%
indices_identity, indices_transpose = data_hidden.calculate_indices()

#%%
layer = SparseMatrixEquivariantLayer(schema, 1, 3).to(device)
#%%
data_out_layer = layer(data_hidden, indices_identity, indices_transpose)
#%%


data_out = net(data_hidden, indices_identity, indices_transpose)

#%%

# Loss functions:
def sparse_loss_fcn(data_pred, data_true):
    loss = torch.zeros(1).to(device)
    for relation in relations:
        rel_id = relation.id
        loss += torch.mean((data_pred[rel_id].values - data_true[rel_id].values)**2)
    loss = loss / len(relations)
    return loss

def dense_loss_fcn(data_pred, data_true):
    # TODO: Compare sparse and dense implementations
    pass

def val_loss_fcn(data_pred, data_true):
    #TODO: Validation set
    pass

learning_rate = 1e-3
optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.0, 0.999))

sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                             factor=0.5,
                                             patience=10,
                                             verbose=True)

#%%
epochs= 3
progress = tqdm(range(epochs), desc="Loss: ", position=0, leave=True)
for i in progress:
    optimizer.zero_grad()
    data_out = net.forward(data_hidden, indices_identity, indices_transpose)
    train_loss = sparse_loss_fcn(data_out, data_hidden)
    train_loss.backward()
    optimizer.step()
    progress.set_description("Train: {:.4f}".format(train_loss.item()))
