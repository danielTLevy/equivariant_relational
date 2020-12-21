#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 22:43:29 2020

@author: Daniel
"""

#%%
import numpy as np
import math
import torch
import torch.nn as nn
from partitions import get_all_input_output_partitions
from DataSchema import DataSchema, Entity, Relation

class EquivariantLayerSingleParam(nn.Module):
    # Layer mapping a single parameter between two relations
    def __init__(self, equality_mapping, output_shape):
        super(EquivariantLayerSingleParam, self).__init__()
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
        for i in range(len(list_of_diagonals)):
            diagonal_dims = list_of_diagonals[i]
            # First, permute all dimensions in diagonal_dims to end of tensor
            permutation = list(range(X.ndimension()))
            for dim in sorted(diagonal_dims, reverse=True):
                permutation.append(permutation.pop(dim))
            X = X.permute(permutation)
            # Next, update list_of_diagonals to reflect this permutation
            
            update_list_of_diagonals = list_of_diagonals.copy()
            for j, input_dims in enumerate(list_of_diagonals):
                update_list_of_diagonals[j] = [permutation.index(input_dim) for input_dim in input_dims]
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
        print(input_diagonals)
        print(list_of_diagonals)
        for i, diag_dims in enumerate(input_diagonals):
            for j, (input_dims, output_dims) in enumerate(self.equality_mapping):
                if set(input_dims) == set(diag_dims):
                    updated_equalities[j] = ({i}, self.equality_mapping[j][1])
        self.equality_mapping = updated_equalities
        return X

    def pool(self, X):
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

class EquivariantLayerBlock(nn.Module):
    # Layer mapping between two relations
    def __init__(self, input_dim, output_dim, data_schema, relation_in, relation_out):
        super(EquivariantLayerBlock, self).__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.input_output_partitions = get_all_input_output_partitions(data_schema, relation_in, relation_out)
        self.params = len(self.input_output_partitions)
        stdv = 1. / math.sqrt(self.in_dim)
        self.weights = nn.Parameter(torch.Tensor(self.params, self.in_dim, self.out_dim).uniform_(-stdv, stdv))
        self.output_shape = tuple(entity.n_instances for entity in relation_out.entities)

    
    def diag(self, X):
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
            for j, input_dims in enumerate(list_of_diagonals):
                update_list_of_diagonals[j] = [permutation.index(input_dim) for input_dim in input_dims]
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
                    updated_equalities[j] = ({i}, self.equality_mapping[j][1])
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
        Y = torch.zeros(self.output_shape)
        for i in range(self.params):
            self.equality_mapping = self.input_output_partitions[i]
            Y_i = self.diag(X)
            Y_i = self.pool(Y_i)
            Y_i = self.broadcast(Y_i)
            Y_i = self.undiag(Y_i)
            Y_i = self.reindex(Y_i)
            Y = Y + (Y_i * self.weights[i].view(-1))
        return Y


#%%
X = torch.tensor(np.arange(12, dtype=np.float32)).view(2,2,3)
# Entity index : number instances mapping
entities = [Entity(0, 3), Entity(1, 2), Entity(2, 5)]
relation_i = Relation([entities[1], entities[1], entities[0]]) #2, 2, 3
relation_j = Relation([entities[0], entities[1]]) #3, 2
relations = [relation_i, relation_j]
data_schema = DataSchema(entities, relations)

equilayer = EquivariantLayerBlock(1, 1, data_schema, relation_i, relation_j)
equilayer.equality_mapping = [({2}, {0}), ({0}, set()), ({1}, {1})]
equilayer.forward(X)