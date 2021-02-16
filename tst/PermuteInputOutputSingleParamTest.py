# -*- coding: utf-8 -*-

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
from DataSchema import Entity, Relation
from EquivariantLayerSingleParam import EquivariantLayerSingleParam
torch.manual_seed(0)

def permute_entities(X_in, X_out, entities, relation_i, relation_j):
    entity_permutations = {}
    for entity in entities:
        permutation = np.random.permutation(entity.n_instances)
        while (permutation == np.arange(entity.n_instances )).all():
            permutation = np.random.permutation(entity.n_instances)
        entity_permutations[entity.id] = permutation
    
    for dim, entity in enumerate(relation_i.entities):
        permutations = [slice(None)]*(dim + 1)
        permutations += [list(entity_permutations[entity.id])]
        permutations += [...]
        X_in = X_in[permutations]
    
    for dim, entity in enumerate(relation_j.entities):
        permutations = [slice(None)]*(dim + 1)
        permutations += [list(entity_permutations[entity.id])]
        permutations += [...]
        X_out = X_out[permutations]
    
    return X_in, X_out

def test_param_without_permutation(X, mapping, output_shape):
    torch.manual_seed(0)
    layer = EquivariantLayerSingleParam(mapping, output_shape)
    return layer.forward(X)
    
def test_param_with_permutation(X, mapping, output_shape, entities, relation_i, relation_j):
    X_exp = test_param_without_permutation(X, mapping, output_shape)
    X_in_perm, X_exp_perm = permute_entities(X, X_exp, entities, relation_i, relation_j)
    torch.manual_seed(0)
    layer = EquivariantLayerSingleParam(mapping, output_shape)
    X_out_perm = layer.forward(X_in_perm)

    return torch.equal(X_out_perm, X_exp_perm)


#%%
if __name__ == '__main__': 
    #Example 
    ##Ri = {n1, n2, n3}
    #Rj = {m1, m2}
    #equalities = {{n1, n2, m2}, {n3}, {m1}}
    X = torch.tensor(np.arange(12)).view(1,2,2,3)
    mapping = [({1,2},{2}), ({3},set()), (set(),{1})]
    entities = [Entity(0, 3), Entity(1, 2), Entity(2, 5)]
    relation_i = Relation(0, [entities[1], entities[1], entities[0]])
    relation_j = Relation(1, [entities[2], entities[1]])

    output_shape = (1, 5, 2)
    assert test_param_with_permutation(X, mapping, output_shape, entities, relation_i, relation_j)
    # Take diagonal of dimensions 0 and 1. Pool result along dimension 2. Broadcast along dimension 0 of output.

    # %%
    # Example 2
    #Ri = {n1, n2}
    #Rj = {m1, m2, m3}
    #equalities = {{n1, n2}, {m2, m3}, {m1}}
    X = torch.tensor(np.arange(16)).view(1, 4,4)
    output_shape = (1, 3, 2, 2)
    mapping = [({1,2},set()), (set(),{2,3}), (set(),{1})]
    entities = [Entity(0, 3), Entity(1, 2), Entity(2, 4)]
    relation_i = Relation(0, [entities[2], entities[2]])
    relation_j = Relation(1, [entities[0], entities[1], entities[1]])
    assert test_param_with_permutation(X, mapping, output_shape, entities, relation_i, relation_j)

    # Take diagonal of dimensions 0 and 1, and pool over the resulting dimension. 
    # Broadcast onto three dimensions. Take the diagonal of resulting dimensions 1 and 2


    # %%
    # Examples 3
    #Ri = {n1, n2, n3}
    #Rj = {m1, m2, m3}
    #equalities = {{n1, n3, m3}, {n2}, {m1, m2}}
    X = torch.tensor(np.arange(18)).view(1, 3, 2, 3)
    mapping = [({1,3},{3}), ({2},set()), (set(),{1,2})]
    output_shape = (1, 4, 4, 3)
    entities = [Entity(0, 3), Entity(1, 2), Entity(2, 4)]
    relation_i = Relation(0, [entities[0], entities[1], entities[0]])
    relation_j = Relation(1, [entities[2], entities[2], entities[0]])

    assert test_param_with_permutation(X, mapping, output_shape, entities, relation_i, relation_j)

    # From input tensor, take diagonal along dimensions 0 and 2 (forming new dimension 0), and pool along dimension 1
    # Broadcast resultant rank 1 tensor onto 3 dimensions, and then take only the diagonal of first two dimensions (leaving last dimension)



    #%%
    # Example 4
    # Ri: n0, n1, n2, n3, n4, n5
    # Rj: m0, m1, m2, m3
    # Equalitiest = {{n1, n3}, {n0, m1}, {m2}, {n2, n4, n5, m0}}
    # Want to take diagonal along 1 and 3 and pool along that (a)
    # Want to transpose 0th dimension to 1st dimension (b)
    # Want broadcast result along second dimension (c)
    # Want to take diagonal along 2nd, 4th, and 5th dimension, and broadcast to 0th dimension


    #            a               b          c               d
    mapping = [({2, 4}, set()), ({1}, {2}), (set(), {3}), ({3, 5, 6}, {1})]
    a = 4
    b = 3
    c = 5
    d = 2
    X = torch.Tensor(np.arange(b*a*d*a*d*d)).view(1, b, a, d, a, d, d)

    entities = [Entity(0, 4), Entity(1, 3), Entity(2, 5), Entity(3, 2)]
    relation_i = Relation(0, [entities[1], entities[0], entities[3],
                           entities[0], entities[3], entities[3]])
    relation_j = Relation(1, [entities[3], entities[1], entities[2]])
    output_shape = (1, d, b, c)
    assert test_param_with_permutation(X, mapping, output_shape, entities, relation_i, relation_j)
    
    
    #%%
    # Example 5:
    # Same as example 2 but with multiple channels
    X = torch.tensor(np.arange(48)).view(3, 4,4)
    output_shape = (3, 3, 2, 2)
    mapping = [({1,2},set()), (set(),{2,3}), (set(),{1})]
    entities = [Entity(0, 3), Entity(1, 2), Entity(2, 4)]
    relation_i = Relation(0, [entities[2], entities[2]])
    relation_j = Relation(1, [entities[0], entities[1], entities[1]])
    assert test_param_with_permutation(X, mapping, output_shape, entities, relation_i, relation_j)