#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 22:43:29 2020

@author: Daniel
"""

#%%
import numpy as np
import torch 
import torch.nn as nn

    
#%%
# TODO:
# Replace this inputsize outputsize thing with number of Ns
# compute all equality_mapping 


class EquivariantLayer(nn.Module):
    # For now, define it as being from Ri to Rj instead of general
    # Required to have the equalities between dimensions 
    def __init__(self, equality_mapping, output_shape):
        super(EquivariantLayer, self).__init__()
        self.equality_mapping = equality_mapping
        self.output_shape = output_shape
    
    def diag(self, X):
        ''' 
        Takes diagonal along any input dimensions that are equal,
        For example, with a a rank 3 tensor X, D = [{0, 2} , {1}] will 
        produce a rank 2 tensor whose 0th dimension is the 1st dimension of X,
        and whose 1st dimension is the diagonal of the 0th and 2nd dimensions
        of X
        '''
        
        # Get dimensions to take diagonals along
        input_diagonals = [sorted(list(i)) for i, o in self.equality_mapping if i != set()]
        list_of_diagonals = input_diagonals.copy()
        # Take diagonal along each dimension
        for i, diagonal_dims in enumerate(list_of_diagonals):
            # First, permute all dimensions in diagonal_dims to end of tensor
            permutation = list(range(X.ndimension()))
            for dim in sorted(diagonal_dims, reverse=True):
                permutation.append(permutation.pop(dim))
            X = X.permute(permutation)
            # Next, update list_of_diagonals to reflect this permutation
            for j, input_dims in enumerate(list_of_diagonals[i:]):
                list_of_diagonals[j] = [permutation.index(input_dim) for input_dim in input_dims]
            
            # Finally, take a diagonal of the last len(diagonal_dims) dimensions
            n_diagonal_dims = len(diagonal_dims)
            if n_diagonal_dims == 1:
                continue
            else:
                for j in range(n_diagonal_dims - 1):
                    X = X.diagonal(0, -1, -2)
        
        # Update dimensions in equality_mapping
        updated_equalities = self.equality_mapping.copy()
        for i, diag_dims in enumerate(input_diagonals):
            for j, (input_dims, output_dims) in enumerate(self.equality_mapping):
                if set(input_dims) == set(diag_dims):
                    updated_equalities[j] = ({i}, self.equality_mapping[j][1])
        self.equality_mapping = updated_equalities
        return X

    def pool(self, X):
        '''
        Sum along any input dimensions that are not in the output
        Update the equality mappings
        '''
    	# Sum along the dimensions listed in P, reducing the number of dimensions in X by sizeof(P)
        pooling_dims = []
        for i, o in self.equality_mapping:
            if o == set():
                pooling_dims += list(i)
        pooling_dims = sorted(pooling_dims)
        for pooling_dim in pooling_dims:
            X = X.sum(pooling_dim)
            # Update equality_mapping by removing pooled dimensions and updating indices
            for index, (input_dims, output_dims) in enumerate(self.equality_mapping):
                if pooling_dim in input_dims:
                    input_dims.remove(pooling_dim)
                input_dims = {input_dim - 1 if input_dim > pooling_dim
                              else input_dim
                              for input_dim in input_dims}
                self.equality_mapping[index] = (input_dims, output_dims)

            pooling_dims = [p - 1 for p in pooling_dims]

        # Remove any empty input-output equalities
        empty_equality = (set(), set())       
        self.equality_mapping = list(filter(
                lambda a: a != empty_equality, self.equality_mapping))

        return X
    
    def broadcast(self, X):
        '''
        Expand X to add a new dimension for every output dimension that does 
        not have a corresponding input dimension
        '''
        for index, (i, o) in enumerate(self.equality_mapping):
            if i != set():
                continue
            
            new_dim_size = self.output_shape[list(o)[0]]
            # Add new dimension of size new_dim_size, and keep rest of dimensions the same
            X = X.expand(new_dim_size, *X.shape)
            # New dimension is now first dimension, increment all dimensions:
            for index2, (i2, o2) in enumerate(self.equality_mapping):
                if i2 != set():
                    self.equality_mapping[index2] = ({i2.pop() + 1}, o2)
            self.equality_mapping[index] = ({0}, o)
        
        return X

    def undiag(self, X):
        '''
        Expand out any dimensions that are equal in the output tensor 
        to their diagonal
        '''
        for index, (i, o)  in enumerate(self.equality_mapping):

            input_dim = list(i)[0]

            # First, permute input_dim to end of tensor
            permutation = list(range(X.ndimension()))
            permutation.append(permutation.pop(input_dim))
            X = X.permute(permutation)
            # Next, update equality_mapping to reflect this permutation
            for index2, (i2, o2) in enumerate(self.equality_mapping):
                self.equality_mapping[index2] = ({permutation.index(dim) for dim in i2}, o2)

            # Then, embed diag it: expand it out to more dimensions
            n_diagonal_dims = len(o)
            if n_diagonal_dims == 1:
                continue
            else:
                for j in range(n_diagonal_dims - 1):
                    X = X.diag_embed(0, -2, -1)
            # update equality mapping so that input is replaced by last n_dim dimensions of tensor
            new_input = np.arange(X.ndim - n_diagonal_dims, X.ndim)
            self.equality_mapping[index] = ({*new_input}, o)
        return X

    def reindex(self, X):
        '''
        Permute the dimensions of X based on the equality mapping
        '''
        output_dims= []
        for i, o in self.equality_mapping:
            output_dims += list(o)
        permutation = [output_dims.index(dim) for dim in np.arange(len(output_dims))]
        X = X.permute(*permutation)
        return X

    def forward(self, X):
        X = self.diag(X)
        X = self.pool(X)
        X = self.broadcast(X)
        X = self.undiag(X)
        X = self.reindex(X)
        return X
