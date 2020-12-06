#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 22:43:29 2020

@author: Daniel
"""

#%%
import numpy as np
import torch 
from EquivariantLayer import EquivariantLayer
    
#%%
def test_layer_output(X, equality_mapping, output_shape, X_expected):
    layer = EquivariantLayer(equality_mapping, output_shape)
    X_out = layer.forward(X)
    return torch.equal(X_out, X_expected)

if __name__ == '__main__': 
    #%%
    #Example 
    ##Ri = {n1, n2, n3}
    #Rj = {m1, m2}
    #equalities = {{n1, n2, m2}, {n3}, {m1}}
    X = torch.tensor(np.arange(12)).view(2,2,3)
    mapping = [({0,1},{1}), ({2},set()), (set(),{0})]
    output_shape = (5, 2)
    X_expected = torch.tensor([[ 3, 30],
            [ 3, 30],
            [ 3, 30],
            [ 3, 30],
            [ 3, 30]])
    assert test_layer_output(X, mapping, output_shape, X_expected)
    # Take diagonal of dimensions 0 and 1. Pool result along dimension 2. Broadcast along dimension 0 of output.

    # %%
    # Example 2
    #Ri = {n1, n2}
    #Rj = {m1, m2, m3}
    #equalities = {{n1, n2}, {m2, m3}, {m1}}
    X = torch.tensor(np.arange(16)).view(4,4)
    output_shape = (3, 2, 2)
    mapping = [({0,1},set()), (set(),{1,2}), (set(),{0})]
    X_expected = torch.tensor([[[30,  0],
             [ 0, 30]],
    
            [[30,  0],
             [ 0, 30]],
    
            [[30,  0],
             [ 0, 30]]])
    assert test_layer_output(X, mapping, output_shape, X_expected)

    # Take diagonal of dimensions 0 and 1, and pool over the resulting dimension. 
    # Broadcast onto three dimensions. Take the diagonal of resulting dimensions 1 and 2

    #%%

    X0 = X = torch.tensor(np.arange(6)).view(2, 3)
    output_shape = (3, 2, 1, 3, 4, 1, 1) 
    ia = [(set(),{4}), ({1},{0,3}), (set(),{2,5,6}), ({0},{1})]
    examplea = EquivariantLayer(ia, output_shape)
    X = examplea.broadcast(X)
    X = examplea.undiag(X)
    X = examplea.reindex(X)
    # %%
    # Examples 3
    #Ri = {n1, n2, n3}
    #Rj = {m1, m2, m3}
    #equalities = {{n1, n3, m3}, {n2}, {m1, m2}}
    X = torch.tensor(np.arange(18)).view(3, 2, 3)
    mapping = [({0,2},{2}), ({1},set()), (set(),{0,1})]
    output_shape = (4, 4, 3)
    X_expected = torch.tensor([[[ 3, 17, 31],
             [ 0,  0,  0],
             [ 0,  0,  0],
             [ 0,  0,  0]],
    
            [[ 0,  0,  0],
             [ 3, 17, 31],
             [ 0,  0,  0],
             [ 0,  0,  0]],
    
            [[ 0,  0,  0],
             [ 0,  0,  0],
             [ 3, 17, 31],
             [ 0,  0,  0]],
    
            [[ 0,  0,  0],
             [ 0,  0,  0],
             [ 0,  0,  0],
             [ 3, 17, 31]]])
    assert test_layer_output(X, mapping, output_shape, X_expected)
    
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
    mapping = [({1, 3}, set()), ({0}, {1}), (set(), {2}), ({2, 4, 5}, {0})]
    a = 4
    b = 3
    c = 5
    d = 2
    X = torch.Tensor(np.arange(b*a*d*a*d*d)).view(b, a, d, a, d, d)
    X_expected = torch.tensor([[[ 16.,  16.,  16.,  16.,  16.],
             [272., 272., 272., 272., 272.],
             [528., 528., 528., 528., 528.]],
            [[ 94.,  94.,  94.,  94.,  94.],
             [350., 350., 350., 350., 350.],
             [606., 606., 606., 606., 606.]]])
    output_shape = (d, b, c)
    assert test_layer_output(X, mapping, output_shape, X_expected)
