# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from src.EquivariantNetwork import EquivariantAutoEncoder
from src.GenerateData import SchoolGenerator

N_STUDENT = 100
N_COURSE = 20
N_PROFESSOR = 5
EMBEDDING_DIMS = 5
BATCH_SIZE = 8
data_generator = SchoolGenerator(N_STUDENT, N_COURSE, N_PROFESSOR)
data = data_generator.generate_data(EMBEDDING_DIMS, BATCH_SIZE)
schema = data_generator.schema
relations = schema.relations

#%%
# Normalize the data
def norm_data(data):
    std_means = {}
    data_normed = {}
    for relation in relations:
        std, mean = torch.std_mean(data[relation.id])
        data_normed[relation.id] = (data[relation.id] - mean) / std
        std_means[relation.id] = (std, mean)
    return data_normed, std_means

def unnorm_data(data, std_means):
    data_out = {}
    for relation in relations:
        std, mean = std_means[relation.id]
        data_out[relation.id] = std*data[relation.id] + mean
    return data_out

data_normed, std_means = norm_data(data)
data_normed_unbatched = {rel.id: data_normed[rel.id][0].unsqueeze(0) for rel in relations}
#%%
# Train the neural network
net = EquivariantAutoEncoder(schema)

#%%
# Loss functions:
def loss_fcn(data_pred, data_true):
    loss = torch.zeros(1)
    for relation in relations:
        rel_id = relation.id
        loss += torch.sum((data_pred[rel_id] - data_true[rel_id])**2)
    loss = loss / len(relations)
    return loss

def per_rel_loss(data_pred, data_true):
    loss = {}
    for relation in relations:
        rel_id = relation.id
        loss[rel_id] = torch.mean((data_pred[rel_id] - data_true[rel_id])**2)
    return loss

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

learning_rate = 1e-4
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)

#%%
epochs=1000
progress = tqdm(range(epochs), desc="Loss: ", position=0, leave=True)
j = 0 
for i in progress:
    optimizer.zero_grad()
    data_out = net.forward(data_normed)
    loss = loss_fcn(data_out, data_normed)
    loss.backward()
    optimizer.step()
    progress.set_description("Loss: {:.4f}".format(loss.item()))

#%%
encoding_size = net.get_encoding_size()
total_encoding_size = sum([enc[0]*enc[1] for enc in encoding_size.values()])
num_els = {key: val.numel() for key, val in data.items()}
print("Total datapoints: ", sum(d.numel() for d in data.values()))
print("Total params: ", sum(p.numel() for p in net.parameters() if p.requires_grad))
losses = per_rel_loss(data_out, data_normed)
#%%
data_unnormed = unnorm_data(data_out, std_means)
#%%
for name, param in net.named_parameters():
    if param.requires_grad:
        print(name, param.data)
