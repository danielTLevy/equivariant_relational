# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from src.EquivariantNetwork import EquivariantAutoEncoder
from src.GenerateData import SchoolGenerator
    
#%%
N_STUDENT = 500
N_COURSE = 20
N_PROFESSOR = 10
EMBEDDING_DIMS = 5
data_generator = SchoolGenerator(N_STUDENT, N_COURSE, N_PROFESSOR)
data = data_generator.generate_data(EMBEDDING_DIMS)
schema = data_generator.schema
relations = schema.relations

#%%
# Train the neural network
net = EquivariantAutoEncoder(schema)


#%%
# Normalize the data
std_means = {}
for relation in relations:
    std, mean = torch.std_mean(data[relation.id])
    data[relation.id] = (data[relation.id] - mean) / std
    std_means[relation.id] = (std, mean)

#%%
def loss_fcn(data_pred, data_true):
    loss = torch.zeros(1)
    for relation in relations:
        rel_id = relation.id
        loss += torch.mean((data_pred[rel_id] - data_true[rel_id])**2) 
    return loss    

learning_rate = 1e-3
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
#%%
epochs=1000
progress = tqdm(range(epochs), desc="Loss: ", position=0, leave=True)
j = 0 
for i in progress:
    optimizer.zero_grad()
    data_out = net.forward(data)
    loss = loss_fcn(data_out, data)
    loss.backward()
    optimizer.step()
    progress.set_description("Loss: {:.4f}".format(loss.item()))
    #if j % 1000 == 0:
    #    val_loss = val_loss_fcn(data, data_train)
    #    print("\nval_loss: ", val_loss.item(), "\n")
    #j = j + 1

#%%

print("Total datapoints: ", sum(d.numel() for d in data.values()))
print("Total params: ", sum(p.numel() for p in net.parameters() if p.requires_grad))
#%%

for name, param in net.named_parameters():
    if param.requires_grad:
        print(name, param.data)