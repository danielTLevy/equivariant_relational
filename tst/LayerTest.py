#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 00:38:08 2020

@author: Daniel
"""
#TODO: Make this work with full layers by making a more full data instead
# of just an X

#%%
import sys
sys.path.append('../src')
import numpy as np
import torch
from DataSchema import DataSchema, Entity, Relation
from EquivariantLayer import EquivariantLayer


def test_layer(X, entities, relation_i, relation_j, in_dim, out_dim):
    torch.manual_seed(0)
    relations = [relation_i, relation_j]
    schema = DataSchema(entities, relations)
    layer = EquivariantLayer(schema, in_dim, out_dim)
    X_out = layer.forward(X)
    print("X_out: ", X_out)
    expected_shape = [out_dim] + [entity.n_instances for entity in relation_j.entities]
    print("Out shape: ", list(X_out.shape))
    print("Expected shape: ", expected_shape)
    assert(list(X_out.shape) == expected_shape)


#%%
# NOTE: THIS CODE CURRENTLY DOES NOT WORK
if __name__ == '__main__': 
    
    #Example 
    ##Ri = {n1, n2, n3}
    #Rj = {m1, m2}
    #equalities = {{n1, n2, m2}, {n3}, {m1}}
    X = torch.tensor(np.arange(12, dtype=np.float32)).view(1, 2,2,3)
    # Entity index : number instances mapping
    entities = [Entity(0, 3), Entity(1, 2), Entity(2, 5)]
    relation_i = Relation(0, [entities[1], entities[1], entities[0]])
    relation_j = Relation(1, [entities[0], entities[1]])
    relation_k = Relation(2 ,[entities[2], entities[2]])
    data_schema = DataSchema(entities, [relation_i, relation_j, relation_k])
    test_layer(X, entities, relation_i, relation_j, 1, 1)

    # %%
    # Example 2
    #Ri = {n1, n2}
    #Rj = {m1, m2, m3}
    #equalities = {{n1, n2}, {m2, m3}, {m1}}
    X = torch.tensor(np.arange(16, dtype=np.float32)).view(1, 4,4)
    entities = [Entity(0, 3), Entity(1, 2), Entity(2, 4)]
    relation_i = Relation(0, [entities[2], entities[2]])
    relation_j = Relation(1, [entities[0], entities[1], entities[1]])
    
    test_layer(X, entities, relation_i, relation_j, 1, 1)
    
    # Take diagonal of dimensions 0 and 1, and pool over the resulting dimension. 
    # Broadcast onto three dimensions. Take the diagonal of resulting dimensions 1 and 2


    # %%
    # Examples 3
    #Ri = {n1, n2, n3}
    #Rj = {m1, m2, m3}
    #equalities = {{n1, n3, m3}, {n2}, {m1, m2}}
    X = torch.tensor(np.arange(18, dtype=np.float32)).view(1, 3, 2, 3)
    entities = [Entity(0, 3), Entity(1, 2), Entity(2, 4)]
    relation_i = Relation(0, [entities[0], entities[1], entities[0]])
    relation_j = Relation(1, [entities[2], entities[2], entities[0]])

    test_layer(X, entities, relation_i, relation_j, 1, 1)
    
    # From input tensor, take diagonal along dimensions 0 and 2 (forming new dimension 0), and pool along dimension 1
    # Broadcast resultant rank 1 tensor onto 3 dimensions, and then take only the diagonal of first two dimensions (leaving last dimension)
    

    #%%
    # Example 4
    # Ri: n0, n1, n2, n3, n4, n5
    # Rj: m0, m1, m2, m3
    # Equalities = {{n1, n3}, {n0, m1}, {m2}, {n2, n4, n5, m0}}
    # Want to take diagonal along 1 and 3 and pool along that (a)
    # Want to transpose 0th dimension to 1st dimension (b)
    # Want broadcast result along second dimension (c)
    # Want to take diagonal along 2nd, 4th, and 5th dimension, and broadcast to 0th dimension
    
    
    #            a               b          c               d
    mapping = [({1, 3}, set()), ({0}, {1}), (set(), {2}), ({2, 4, 5}, {0})]
    a = 4
    b = 3
    c = 5
    d = 2
    X = torch.Tensor(np.arange(b*a*d*a*d*d)).view(1, b, a, d, a, d, d)
    entities = [Entity(0, 4), Entity(1, 3), Entity(2, 5), Entity(3, 2)]
    relation_i = Relation(0, [entities[1], entities[0], entities[3],
                           entities[0], entities[3], entities[3]])
    relation_j = Relation(1, [entities[3], entities[1], entities[2]])
    test_layer(X, entities, relation_i, relation_j, 1, 1)
