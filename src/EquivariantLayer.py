#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 22:43:29 2020

@author: Daniel
"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from src.utils import get_all_input_output_partitions, PREFIX_DIMS
from src.DataSchema import Data
import pdb

class EquivariantLayerBlock(nn.Module):
    # Layer mapping between two relations
    def __init__(self, input_dim, output_dim, relation_in, relation_out):
        super(EquivariantLayerBlock, self).__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.input_output_partitions = get_all_input_output_partitions(relation_in, relation_out)
        self.n_params = len(self.input_output_partitions)
        stdv = 0.1 / math.sqrt(self.in_dim)
        self.weights = nn.Parameter(torch.Tensor(self.n_params, self.in_dim, self.out_dim).uniform_(-stdv, stdv))
        self.bias = nn.Parameter(torch.ones(self.n_params))
        self.output_shape = [0, self.out_dim] + [entity.n_instances for entity in relation_out.entities]


    def diag(self, X, equality_mapping):
        '''
        #Takes diagonal along any input dimensions that are equal,
        #For example, with a a rank 3 tensor X, D = [{0, 2} , {1}] will 
        #produce a rank 2 tensor whose 0th dimension is the 1st dimension of X,
        #and whose 1st dimension is the diagonal of the 0th and 2nd dimensions
        #of X
        '''
        
        # Get dimensions to take diagonals along
        input_diagonals = [sorted(list(i)) for i, o in equality_mapping if i != set()]
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
        updated_equality_mapping = equality_mapping.copy()
        for i, diag_dims in enumerate(input_diagonals):
            for j, (input_dims, output_dims) in enumerate(equality_mapping):
                if set(input_dims) == set(diag_dims):
                    updated_equality_mapping[j] = ({PREFIX_DIMS+i}, equality_mapping[j][1])
        return X, updated_equality_mapping


    def pool(self, X, equality_mapping):
        '''
        Sum along any input dimensions that are not in the output
        Update the equality mappings
        '''
        # Pool over dimensions with no correponding output
        index_array = np.arange(len(X.shape))
        pooling_dims = []
        for i, o in equality_mapping:
            if o == set():
                pooling_dims += list(i)
        pooling_dims = sorted(pooling_dims)
        if pooling_dims != []:
            # TODO: can make this a max
            #X = X.amax(pooling_dims)
            X = X.mean(pooling_dims)
    
            # Get updated indices
            index_array = np.delete(index_array, pooling_dims)
            update_mapping = {value: index for index, value in enumerate(index_array)}
    
            # Replace equality mapping with new indices
            updated_equality_mapping = []
            for i, o in equality_mapping:
                if o == set():
                    continue
                i_new = set()
                for el in i:
                    i_new.add(update_mapping[el])
                updated_equality_mapping.append((i_new, o))
        else:
            updated_equality_mapping = equality_mapping.copy()

        return X, updated_equality_mapping

    def broadcast(self, X, output_shape, equality_mapping):
        '''
        Expand X to add a new dimension for every output dimension that does 
        not have a corresponding input dimension
        '''
        updated_equality_mapping = []
        for index, (i, o) in enumerate(equality_mapping):
            if i != set():
                updated_equality_mapping.append((i,o))
                continue
            
            new_dim_size = output_shape[list(o)[0]]
            X = X.unsqueeze(-1).expand(*X.shape, new_dim_size)
            updated_equality_mapping.append(({X.ndimension() - 1}, o))

        return X, updated_equality_mapping

    def undiag(self, X, equality_mapping):
        '''
        Expand out any dimensions that are equal in the output tensor 
        to their diagonal
        '''
        updated_equality_mapping = equality_mapping.copy()
        for index, (i, o)  in enumerate(updated_equality_mapping):

            input_dim = list(i)[0]

            # First, permute input_dim to end of tensor
            permutation = list(range(X.ndimension()))
            permutation.append(permutation.pop(input_dim))
            X = X.permute(permutation)
            # Next, update equality_mapping to reflect this permutation
            for index2, (i2, o2) in enumerate(updated_equality_mapping):
                updated_equality_mapping[index2] = ({permutation.index(dim) for dim in i2}, o2)

            # Then, embed diag it: expand it out to more dimensions
            n_diagonal_dims = len(o)
            if n_diagonal_dims == 1:
                continue
            else:
                for j in range(n_diagonal_dims - 1):
                    X = X.diag_embed(0, -2, -1)
            # update equality mapping so that input is replaced by last n_dim dimensions of tensor
            new_input = np.arange(X.ndim - n_diagonal_dims, X.ndim)
            updated_equality_mapping[index] = ({*new_input}, o)
        return X, updated_equality_mapping

    def reindex(self, X, equality_mapping):
        '''
        Permute the dimensions of X based on the equality mapping
        '''
        output_dims= []
        for i, o in equality_mapping:
            output_dims += list(o)
        permutation = [0, 1] + [PREFIX_DIMS + output_dims.index(PREFIX_DIMS+dim)
                                    for dim in np.arange(len(output_dims))]
        X = X.permute(permutation)
        return X, equality_mapping



    def forward(self, X):
        Y = None

        for i in range(self.n_params):
            equality_mapping = self.input_output_partitions[i]

            Y_i, equality_mapping = self.diag(X, equality_mapping)

            Y_i, equality_mapping = self.pool(Y_i, equality_mapping)

            Y_i, equality_mapping = self.broadcast(Y_i, self.output_shape, equality_mapping)

            Y_i, equality_mapping = self.undiag(Y_i, equality_mapping)

            Y_i, equality_mapping = self.reindex(Y_i, equality_mapping)

            Y_i = F.linear(Y_i.transpose(1,-1), self.weights[i].T).transpose(1,-1) + self.bias[i]
            if Y == None:
                Y = Y_i
            else:
                Y  += Y_i

        return Y


class EquivariantLayer(nn.Module):
    def __init__(self, data_schema, input_dim=1, output_dim=1):
        super(EquivariantLayer, self).__init__()
        self.data_schema = data_schema
        self.relation_pairs = list(itertools.product(self.data_schema.relations,
                                                self.data_schema.relations))
        block_modules = []
        self.input_dim = input_dim
        self.output_dim = output_dim
        for relation_i, relation_j in self.relation_pairs:
            block_module = EquivariantLayerBlock(self.input_dim, self.output_dim,
                                                 relation_i, relation_j)
            block_modules.append(block_module)
        self.block_modules = nn.ModuleList(block_modules)

    def forward(self, data):
        data_out = Data(self.data_schema)
        for i, (relation_i, relation_j) in enumerate(self.relation_pairs):
            X = data[relation_i.id]
            layer = self.block_modules[i]
            out = layer.forward(X)
            if relation_j.id not in data_out:
                data_out[relation_j.id] = out
            else:
                data_out[relation_j.id] = data_out[relation_j.id] + out
        return data_out
