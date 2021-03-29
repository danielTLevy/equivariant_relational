#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from src.utils import get_all_input_output_partitions, PREFIX_DIMS
from src.DataSchema import Data, DataSchema, Entity, Relation
from src.SparseTensor import SparseTensor
import pdb

class SparseEquivariantLayerBlock(nn.Module):
    # Layer mapping between two relations
    def __init__(self, input_dim, output_dim, data_schema, relation_in, relation_out):
        super(SparseEquivariantLayerBlock, self).__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.input_output_partitions = get_all_input_output_partitions(data_schema, relation_in, relation_out)
        self.n_params = len(self.input_output_partitions)
        stdv = 0.1 / math.sqrt(self.in_dim)
        self.weights = nn.Parameter(torch.Tensor(self.n_params, self.in_dim, self.out_dim).uniform_(-stdv, stdv))
        self.bias = nn.Parameter(torch.ones(1))
        self.output_shape = [0, self.out_dim] + [entity.n_instances for entity in relation_out.entities]

    
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
                    updated_equalities[j] = ({PREFIX_DIMS+i}, self.equality_mapping[j][1])
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
            # TODO: can make this a max or mean
            X = X.pool(pooling_dims)
    
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

    def broadcast(self, X, X_out):
        '''
        Expand X to add a new dimension for every output dimension that does 
        not have a corresponding input dimension
        '''
        for index, (i, o) in enumerate(self.equality_mapping):
            if i != set():
                continue
            
            new_dim_size = X_out.shape[list(o)[0]]
            X = X.broadcast(X_out.indices, X_out.indices, new_dim_size)
            self.equality_mapping[index] = ({X.ndimension() - 1}, o)

        return X

    def undiag(self, X, X_out):
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
        permutation = [0, 1] + [PREFIX_DIMS + output_dims.index(PREFIX_DIMS+dim)
                                    for dim in np.arange(len(output_dims))]
        X = X.permute(*permutation)
        return X

    def forward(self, X_in, Y_in):
        Y = None
        for i in range(self.n_params):
            self.equality_mapping = self.input_output_partitions[i]
            Y_out = self.diag(X_in)
            Y_out = self.pool(Y_out)
            pdb.set_trace()
            Y_out = self.broadcast(Y_out, Y_in)
            Y_out = self.undiag(Y_out, Y_in)
            Y_out = self.reindex(Y_out)
            
            weight_i = self.weights[i]
            Y_out = F.linear(Y_out.transpose(1,-1), weight_i.T).transpose(1,-1)
            if Y == None:
                Y = Y_out
            else:
                Y  += Y_out
            
        Y  = Y + self.bias
        return Y


class SparseEquivariantLayer(nn.Module):
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
                                                 data_schema, relation_i, relation_j)
            block_modules.append(block_module)
        self.block_modules = nn.ModuleList(block_modules)

    def forward(self, data):
        data_out = Data(self.data_schema)
        for i, (relation_i, relation_j) in enumerate(self.relation_pairs):
            X_in = data[relation_i.id]
            Y_in = data[relation_j.id]
            layer = self.block_modules[i]
            Y_out = layer.forward(X_in, Y_in)
            if relation_j.id not in data_out:
                data_out[relation_j.id] = Y_out
            else:
                data_out[relation_j.id] = data_out[relation_j.id] + Y_out
        return data_out

class TestSparseLayer(nn.Module):
    def __init__(self):
        super(TestSparseLayer, self).__init__()
        stdv = 0.1 / math.sqrt(1)
        self.weight = nn.Parameter(torch.Tensor(1, 1).uniform_(-stdv, stdv))

    def forward(self, data):
        data = F.linear(data.transpose(1,-1), self.weight.T).transpose(1,-1)

