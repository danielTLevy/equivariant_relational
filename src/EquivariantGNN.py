#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import torch
import torch.nn as nn
import itertools
from partitions import get_all_input_output_partitions
from DataSchema import DataSchema, Relation, Entity

class EquivariantGNNLayer(nn.Module):
    # Layer mapping between two relations of the same order
    def __init__(self, order, n_instances, input_dim, output_dim):
        super(EquivariantGNNLayer, self).__init__()
        self.order = order
        self.in_dim = input_dim
        self.out_dim = output_dim
        # TODO: this parameter shouldn't be required.
        self.n_instances = n_instances
        
        # Create a "schema" of only one relation
        self.entity = Entity(0, self.n_instances)
        self.relation = Relation(0, [self.entity]*order)
        self.data_schema = DataSchema([self.entity], [self.relation])

        # Create all associated equality partitions and corresponding params
        self.input_output_partitions = get_all_input_output_partitions(self.data_schema, self.relation, self.relation)
        self.n_params = len(self.input_output_partitions)
        stdv = 0.1 / math.sqrt(self.in_dim)
        self.weights = nn.Parameter(torch.Tensor(self.n_params, self.in_dim, self.out_dim).uniform_(-stdv, stdv))
        self.bias = nn.Parameter(torch.ones(1))

        self.output_shape = [self.out_dim, self.n_instances]

    
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
            # Add new dimension of size new_dim_size, and keep rest of dimensions the same
            X = X.expand(new_dim_size, *X.shape)
            # Permute this new dimension to be the last dimension
            permutation = list(range(X.ndimension()))
            permutation.append(permutation.pop(0))
            X = X.permute(permutation)
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
        Y = torch.zeros(self.output_shape)
        for i in range(self.params):
            self.equality_mapping = self.input_output_partitions[i]

            Y_i = self.diag(X)
            Y_i = self.pool(Y_i)
            Y_i = self.broadcast(Y_i)
            Y_i = self.undiag(Y_i)
            Y_i = self.reindex(Y_i)
            
            weight_i = self.weights[i]

            Y = Y + (torch.tensordot(Y_i.T,  weight_i, 1).T)
            
        Y = Y + self.bias
        return Y

class GNNPooling(nn.Module):
    '''
    Produce encodings for every instance of every entity
    Encodings for each entity are produced by summing over each relation which
    contains the entity
    '''
    def __init__(self, order, n_instances, dims):
        super(GNNPooling, self).__init__()
        self.order = order
        self.n_instances = n_instances
        self.dims = dims
    
    def pool_tensor(self, X, dim):
        # Pool over every dimension except  specified dimension and channel
        pooling_dims = [i+1 for i in range(self.order) if i != dim]
        return torch.sum(X, pooling_dims)

    def forward(self, data):
        entity_out = torch.zeros(self.dims, self.n_instances)
        for dim in range(self.order):
            entity_out += self.pool_tensor(data, dim)
        return  entity_out

class GNNBroadcasting(nn.Module):
    '''
    Given encodings for each entity, returns tensors in the shape of the 
    given relationships in the schema
    '''
    def __init__(self, order, dims):
        super(GNNBroadcasting, self).__init__()
        self.order = order
        self.dims = dims

    def forward(self, encodings):
        out = torch.zeros([self.dims, self.order])
        num_new_dims = self.order - 1
        for entity_idx, entity in enumerate(relation.entities):
            entity_enc = encodings[entity.id]
            # Expand tensor to appropriate number of dimensions
            for _ in range(num_new_dims):
                entity_enc = entity_enc.unsqueeze(-1)
            # Transpose the entity to the appropriate dimension
            entity_enc.transpose_(1, entity_idx+1)
            # Broadcast-add to output
            out += entity_enc
        return out
    
class EquivariantAutoEncoder(nn.Module):
    def __init__(self, order, n_instances, n_channels):
        super(EquivariantAutoEncoder, self).__init__()
        self.order = order
        self.n_instances = n_instances
        self.n_channels = n_channels
        self.hidden_dim = 10

        self.encoder = nn.Sequential(
                EquivariantGNNLayer(self.order, n_instances, n_channels, 16),
                nn.ReLU(),
                nn.GroupNorm(16,16, affine=False),

                EquivariantGNNLayer(self.order, n_instances, 16, self.hidden_dim),
                nn.GroupNorm(self.hidden_dim, self.hidden_dim, affine=True),
                GNNPooling(order, self.hidden_dim)
                )
        self.decoder = nn.Sequential(
                GNNBroadcasting(order, self.hidden_dim),
                EquivariantGNNLayer(self.order, n_instances, self.hidden_dim, 16),
                nn.ReLU(),
                nn.GroupNorm(16,16, affine=False),
                EquivariantGNNLayer(self.order, n_instances, 16, 1),
                )

        
    def forward(self, data):
        enc  = self.encoder(data)
        out = self.decoder(enc)
        return out  
    
