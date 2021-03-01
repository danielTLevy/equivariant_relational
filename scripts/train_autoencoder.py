# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from src.EquivariantNetwork import EquivariantAutoEncoder
from src.GenerateData import SchoolGenerator, SyntheticData

N_STUDENT = 200
N_COURSE = 200
N_PROFESSOR = 200
EMBEDDING_DIMS = 5
BATCH_SIZE = 1

data_generator = SyntheticData((N_STUDENT, N_COURSE, N_PROFESSOR), sparsity=0.5, embedding_dims=5)
data = data_generator.data
observed = data_generator.observed
missing = {key:  ~val for key, val in observed.items()}
schema = data.schema
relations = schema.relations

#%%
# Normalize the data and hide unobserved
data = data.normalize_data()
data_hidden = data.mask_data(observed)

#%%
# Train the neural network
net = EquivariantAutoEncoder(schema)

#%%
# Loss functions:
def loss_fcn(data_pred, data_true, indices):
    loss = torch.zeros(1)
    for relation in relations:
        rel_id = relation.id
        data_pred_rel = indices[rel_id]*data_pred[rel_id]
        data_true_rel = indices[rel_id]*data_true[rel_id]
        loss += torch.mean((data_pred_rel - data_true_rel)**2)
    loss = loss / len(relations)
    return loss

def per_rel_loss(data_pred, data_true):
    loss = {}
    for relation in relations:
        rel_id = relation.id
        loss[rel_id] = torch.mean((data_pred[rel_id] - data_true[rel_id])**2)
    return loss


learning_rate = 1e-4
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

#%%
epochs=1000
progress = tqdm(range(epochs), desc="Loss: ", position=0, leave=True)
j = 0 
for i in progress:
    optimizer.zero_grad()
    data_out = net.forward(data_hidden)
    train_loss = loss_fcn(data_out, data_hidden, observed)
    val_loss = loss_fcn(data_out, data, missing)
    train_loss.backward()
    optimizer.step()
    progress.set_description("Train: {:.4f}, Val: {:.4f}".format(
            train_loss.item(), val_loss.item()))
    
    

#%%
encoding_size = net.get_encoding_size()
total_encoding_size = sum([enc[0]*enc[1] for enc in encoding_size.values()])
num_els = {key: val.numel() for key, val in data.items()}
print("Total datapoints: ", sum(d.numel() for d in data.values()))
print("Total params: ", sum(p.numel() for p in net.parameters() if p.requires_grad))

def std_mean_per_entity(data_pred, data_true):
    '''For each entity instance in each relation, get the difference between
    the predicted and actual means. Return mean and std of these differences
    '''
    std_mean_diffs = {}
    for relation in relations:
        std_mean_diffs[relation.id] = {}
        for entity_i, entity in enumerate(relation.entities):
            dims = list(set(range(1+len(relation.entities))) -  {1+entity_i})
            true_mean = data_true[relation.id].mean(dims)
            pred_mean = data_pred[relation.id].mean(dims)
            std_mean_diff = torch.std_mean(abs(true_mean - pred_mean))
            std_mean_diffs[relation.id][entity_i] = std_mean_diff
    return std_mean_diffs

for name, param in net.named_parameters():
    if param.requires_grad:
        print(name, param.data)
