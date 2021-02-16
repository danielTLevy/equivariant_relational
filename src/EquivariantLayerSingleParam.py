#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 23:45:52 2021

@author: Daniel
"""

import numpy as np
import torch.nn as nn


class EquivariantLayerSingleParam(nn.Module):
    # Layer mapping a single parameter between two relations
    def __init__(self, equality_mapping, output_shape):
        super(EquivariantLayerSingleParam, self).__init__()
        self.equality_mapping = equality_mapping
        self.output_shape = output_shape
   

    def diag(self, X):
        '''
        #Takes diagonal along any input dimensions that are equal,
        #For example, with a a rank 3 tensor X, D = [{0, 2} , {1}] will 
        #produce a rank 2 tensor whose 0th dimension is the 1st dimension of X,
        #and whose 1st dimension is the diagonal of the 0th and 2nd dimensions
        #of X
        '''
        
        # Get dimensions to take diagonals along
        input_diagonals = [sorted(list(i)) for i, o in self.equality_mapping if i != set()]
        list_of_diagonals = input_diagonals.copy()
        # Take diagonal along each dimension
        for i in range(len(list_of_diagonals)):
            diagonal_dims = list_of_diagonals[i]
            # First, permute all dimensions in diagonal_dims to end of tensor
            permutation = list(range(X.ndimension()))
            for dim in sorted(diagonal_dims, reverse=True):
                permutation.append(permutation.pop(dim))
            X = X.permute(permutation)
            # Next, update list_of_diagonals to reflect this permutation
            
            update_list_of_diagonals = list_of_diagonals.copy()
            for j, input_dims in enumerate(list_of_diagonals[i:]):
                update_list_of_diagonals[i+j] = [permutation.index(input_dim) for input_dim in input_dims]
            list_of_diagonals = update_list_of_diagonals
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
                    updated_equalities[j] = ({1+i}, self.equality_mapping[j][1])
        self.equality_mapping = updated_equalities
        return X


    def pool(self, X):
        '''
        Sum along any input dimensions that are not in the output
        Update the equality mappings
        '''
        # Pool over dimensions with no correponding output
        index_array = np.arange(len(X.shape))
        pooling_dims = []
        for i, o in self.equality_mapping:
            if o == set():
                pooling_dims += list(i)
        pooling_dims = sorted(pooling_dims)
        if pooling_dims != []:
            X = X.sum(pooling_dims)
    
            # Get updated indices
            index_array = np.delete(index_array, pooling_dims)
            update_mapping = {value: index for index, value in enumerate(index_array)}
    
            # Replace equality mapping with new indices
            new_equality_mapping = []
            for i, o in self.equality_mapping:
                if o == set():
                    continue
                i_new = set()
                for el in i:
                    i_new.add(update_mapping[el])
                new_equality_mapping.append((i_new, o))
            self.equality_mapping = new_equality_mapping
        
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
            X = X.unsqueeze(-1).expand(*X.shape, new_dim_size)
            self.equality_mapping[index] = ({X.ndimension() - 1}, o)

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
        permutation = [0] + [1 + output_dims.index(1+dim) for dim in np.arange(len(output_dims))]
        X = X.permute(*permutation)
        return X

    def forward(self, X):
        #print("Initial")
        #print(self.equality_mapping)
        #print(X.shape)
    
        X = self.diag(X)
        #print("Diag")
        #print(self.equality_mapping)
        #print(X.shape)

        X = self.pool(X)
        #print("Pool")
        #print(self.equality_mapping)
        #print(X.shape)

        X = self.broadcast(X)
        #print("Broadcast")
        #print(self.equality_mapping)
        #print(X.shape)

        X = self.undiag(X)
        #print("Undiag")
        #print(self.equality_mapping)
        #print(X.shape)

        X = self.reindex(X)
        #print("Reindex")
        #print(self.equality_mapping)
        #print(X.shape)
        return X
