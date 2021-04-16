#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 22:18:39 2020

@author: Daniel
"""


#%%
import sys
sys.path.append('../src')
import numpy as np
import torch
from src.DataSchema import Entity, Relation
from src.EquivariantLayer import EquivariantLayerBlock


def test_layer_single_block(X, entities, relation_i, relation_j, in_dim, out_dim, batch_size):
    torch.manual_seed(0)
    layer = EquivariantLayerBlock(in_dim, out_dim, relation_i, relation_j)
    X_out = layer.forward(X)
    #print("X_out: ", X_out)
    expected_shape = [batch_size] + [out_dim] + relation_j.get_shape()
    print("Out shape: ", list(X_out.shape))
    print("Expected shape: ", expected_shape)
    assert(list(X_out.shape) == expected_shape)


#%%
if __name__ == '__main__': 
    #Example 
    ##Ri = {n1, n2, n3}
    #Rj = {m1, m2}
    X = torch.tensor(np.arange(12, dtype=np.float32)).view(1, 1, 2,2,3)
    # Entity index : number instances mapping
    entities = [Entity(0, 3), Entity(1, 2), Entity(2, 5)]
    relation_i = Relation(0, [entities[1], entities[1], entities[0]])
    relation_j = Relation(1, [entities[0], entities[1]])

    test_layer_single_block(X, entities, relation_i, relation_j, 1, 7, 1)

    # %%
    # Example 2
    #Ri = {n1, n2}
    #Rj = {m1, m2, m3}
    X = torch.tensor(np.arange(16,  dtype=np.float32)).view(1, 1,4,4)
    entities = [Entity(0, 3), Entity(1, 2), Entity(2, 4)]
    relation_i = Relation(0, [entities[2], entities[2]])
    relation_j = Relation(1, [entities[0], entities[1], entities[1]])
    
    test_layer_single_block(X, entities, relation_i, relation_j, 1, 3, 1)
    
    # Take diagonal of dimensions 0 and 1, and pool over the resulting dimension. 
    # Broadcast onto three dimensions. Take the diagonal of resulting dimensions 1 and 2


    # %%
    # Examples 3
    #Ri = {n1, n2, n3}
    #Rj = {m1, m2, m3}
    X = torch.tensor(np.arange(18, dtype=np.float32)).view(1, 1, 3, 2, 3)
    entities = [Entity(0, 3), Entity(1, 2), Entity(2, 4)]
    relation_i = Relation(0, [entities[0], entities[1], entities[0]])
    relation_j = Relation(1, [entities[2], entities[2], entities[0]])

    test_layer_single_block(X, entities, relation_i, relation_j, 1, 1, 1)
    
    # From input tensor, take diagonal along dimensions 0 and 2 (forming new dimension 0), and pool along dimension 1
    # Broadcast resultant rank 1 tensor onto 3 dimensions, and then take only the diagonal of first two dimensions (leaving last dimension)
    

    #%%
    # Example 4
    # Ri: n0, n1, n2, n3, n4, n5
    # Rj: m0, m1, m2, m3
    #            a               b          c               d
    a = 4
    b = 3
    c = 5
    d = 2
    X = torch.Tensor(np.arange(b*a*d*a*d*d, dtype=np.float32)).view(1, 1, b, a, d, a, d, d)
    entities = [Entity(0, 4), Entity(1, 3), Entity(2, 5), Entity(3, 2)]
    relation_i = Relation(0, [entities[1], entities[0], entities[3],
                           entities[0], entities[3], entities[3]])
    relation_j = Relation(1, [entities[3], entities[1], entities[2]])
    test_layer_single_block(X, entities, relation_i, relation_j, 1, 1, 1)


    # %%
    # Example 5
    # Repeat of example 2 but with 3 channels going to 5, and 2 batches
    X = torch.tensor(np.arange(96, dtype=np.float32)).view(2,3,4,4)
    entities = [Entity(0, 3), Entity(1, 2), Entity(2, 4)]
    relation_i = Relation(0, [entities[2], entities[2]])
    relation_j = Relation(1, [entities[0], entities[1], entities[1]])
    
    test_layer_single_block(X, entities, relation_i, relation_j, 3, 5, 2)
    
    # Take diagonal of dimensions 0 and 1, and pool over the resulting dimension. 
    # Broadcast onto three dimensions. Take the diagonal of resulting dimensions 1 and 2
