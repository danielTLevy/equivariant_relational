#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 22:35:57 2021

@author: Daniel
"""
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from EquivariantNetwork import EquivariantNetwork
from GenerateData import SyntheticData
    
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_STUDENT = 200
N_COURSE = 200
N_PROFESSOR = 200
EMBEDDING_DIMS = 2
BATCH_SIZE = 1

data_generator = SyntheticData((N_STUDENT, N_COURSE, N_PROFESSOR),
                               sparsity=0.5,  embedding_dims=EMBEDDING_DIMS).to(device)
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
# TODO: Get subset for training, subset for test, subset for validation, subset for full test
#data_train = {i: torch.zeros(data_full[i].shape) for i in range(7)}
#for i in range(7):
#    for j in range(1000):
#        train_idx = np.random.choice
#        data_train[j] = data_full[j]


#%%
# Sparsify
for i in range(7):
    test_course_idx = np.random.choice(N_COURSE, 10, replace=False)
    data_train = {i: data_full[i].clone() for i in range(7)}
    data_train[5][..., test_course_idx] = 0#np.nan


#%%
# Try to predict 100 courses
for i in range(7):
    test_course_idx = np.random.choice(N_COURSE, 10, replace=False)
    data_train = {i: data_full[i].clone() for i in range(7)}
    data_train[5][..., test_course_idx] = 0#np.nan


#%%
# Train the neural network
net = EquivariantNetwork(schema)

filled_idx = {relation.id:
            #torch.nonzero(~torch.isnan(data_train[relation.id]), as_tuple=True)
            torch.nonzero(data_train[relation.id], as_tuple=True)
            for relation in relations}


#%%
# Normalize the data
std_means = {}
for i in range(7):
    std, mean = torch.std_mean(data_full[i])
    data_full[i] = (data_full[i] - mean) / std
    data_train[i] = (data_train[i] - mean) / std
    std_means[i] = (std, mean)
#%%
def loss_fcn(data_pred, data_true):
    loss = torch.zeros(1)
    for relation in relations:
        rel_id = relation.id
        loss += torch.mean((data_pred[rel_id][filled_idx[rel_id]] - data_true[rel_id][filled_idx[rel_id]])**2) 
    return loss    

def val_loss_fcn(data_pred, data_true):
    loss = torch.mean((data_pred[5][..., test_course_idx] - data_true[5][..., test_course_idx])**2)
    return loss    
learning_rate = 1e-3
optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.0, 0.999))

#%%
epochs=1000
progress = tqdm(range(epochs), desc="Loss: ", position=0, leave=True)
j = 0 
for i in progress:
    optimizer.zero_grad()
    data = net.forward(data_train)
    loss = loss_fcn(data, data_train)
    loss.backward()
    optimizer.step()
    progress.set_description("Loss: {:.4f}".format(loss.item()))
    if j % 1000 == 0:
        val_loss = val_loss_fcn(data, data_train)
        print("\nval_loss: ", val_loss.item(), "\n")
    j = j + 1

#%%

print("Total datapoints: ", sum(d.numel() for d in data_train.values()))
print("Total params: ", sum(p.numel() for p in net.parameters() if p.requires_grad))
#%%

for name, param in net.named_parameters():
    if param.requires_grad:
        print(name, param.data)