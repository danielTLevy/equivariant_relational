#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 23:25:39 2021

@author: Daniel
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.DataSchema import Data, DataSchema, Relation
from src.utils import PREFIX_DIMS
from src.SparseTensor import SparseTensor
import pdb


class SparseGroupNorm(nn.Module):
    '''
    Normalize each channel separately
    '''
    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True):
        super(SparseGroupNorm, self).__init__()
        self.eps = eps
        self.num_groups = num_groups
        self.num_channels = num_channels
        assert self.num_groups == self.num_channels, "Currently only implemented for num_groups == num_channels"
        self.gamma = nn.Parameter(torch.ones(self.num_channels))
        self.affine = affine
        if self.affine:
            self.beta = nn.Parameter(torch.zeros(self.num_channels))
    
    def forward(self, sparse_tensor):
        values = sparse_tensor.values
        var, mean = torch.var_mean(values, dim=1, unbiased=False)
        values_out = (self.gamma * (values - mean.unsqueeze(1)) \
                      / torch.sqrt(var + self.eps).unsqueeze(1))
        if self.affine:
            values_out += self.beta
        out = sparse_tensor.clone()
        out.values = values_out
        return out

class RelationNorm(nn.Module):
    '''
    Normalize each channel of each relation separately
    '''
    def __init__(self, schema, num_channels, affine, sparse=False):
        super(RelationNorm, self).__init__()
        self.schema = schema
        self.rel_norms = {}
        if sparse:
            GroupNorm = SparseGroupNorm
        else:
            GroupNorm = nn.GroupNorm

        for relation in schema.relations:
            rel_norm = GroupNorm(num_channels, num_channels, affine=affine)
            self.rel_norms[relation.id] = rel_norm

    def forward(self, X):
        for relation in self.schema.relations:
            rel_norm = self.rel_norms[relation.id]
            X[relation.id] = rel_norm(X[relation.id])
        return X

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        for relation in self.schema.relations:
            self.rel_norms[relation.id] = self.rel_norms[relation.id].to(*args, **kwargs)
        return self

class Activation(nn.Module):
    '''
    Extend torch.nn.module modules to be applied to each relation
    '''
    def __init__(self, schema, activation):
        super(Activation, self).__init__()
        self.schema = schema
        self.activation = activation

    def forward(self, X):
        for relation in self.schema.relations:
            X[relation.id] = self.activation(X[relation.id])
        return X

def functional(function, schema, X, *args, **kwargs):
    '''
    Extend torch.nn.functional functions to be applied to each relation
    '''
    for relation in schema.relations:
        X[relation.id] = function(X[relation.id], *args, **kwargs)
    return X


class ReLU(nn.Module):
    def __init__(self, schema):
        super(ReLU, self).__init__()
        self.schema = schema
    
    def forward(self, X):
        for relation in self.schema.relations:
            X[relation.id] = F.relu(X[relation.id]) 
        return X


class EntityBroadcasting(nn.Module):
    '''
    Given encodings for each entity, returns tensors in the shape of the 
    given relationships in the schema
    '''
    def __init__(self, schema, dims):
        super(EntityBroadcasting, self).__init__()
        self.schema = schema
        self.dims = dims
        # empty param just for holding device info
        self.device_param = nn.Parameter(torch.Tensor(0))

    def make_relation(self, encodings, relation):
        batch_size = encodings.batch_size
        relation_shape = [batch_size, self.dims] + relation.get_shape()
        relation_out = torch.zeros(relation_shape, device=self.device_param.device)
        num_new_dims = len(relation.entities) -1
        for entity_idx, entity in enumerate(relation.entities):
            entity_enc = encodings[entity.id]
            # Expand tensor to appropriate number of dimensions
            for _ in range(num_new_dims):
                entity_enc = entity_enc.unsqueeze(-1)
            # Transpose the entity to the appropriate dimension
            entity_enc.transpose_(PREFIX_DIMS, entity_idx+PREFIX_DIMS)
            # Broadcast-add to output
            relation_out += entity_enc
        return relation_out

    def forward(self, encodings):
        data_out = Data(self.schema)
        for relation in self.schema.relations:
            data_out[relation.id] = self.make_relation(encodings, relation)
        return data_out

class EntityPooling(nn.Module):
    '''
    Produce encodings for every instance of every entity
    Encodings for each entity are produced by summing over each relation which
    contains the entity
    '''
    def __init__(self, schema, dims):
        super(EntityPooling, self).__init__()
        self.schema = schema
        self.dims = dims
        self.out_shape = [e.n_instances for e in self.schema.entities]
        # Make a "schema" for the encodings
        enc_relations = [Relation(i, [self.schema.entities[i]])
                            for i in range(len(self.schema.entities))]
        self.enc_schema = DataSchema(self.schema.entities, enc_relations)
    
    def get_pooling_dims(self, entity, relation):
        pooling_dims = []
        for entity_dim, rel_entity in enumerate(relation.entities):
            if entity != rel_entity:
                pooling_dims += [PREFIX_DIMS + entity_dim]
        return pooling_dims
    
    def pool_tensor(self, X, pooling_dims):
        if len(pooling_dims) > 0:
            return torch.sum(X, pooling_dims)
        else:
            return X

    def pool_tensor_diag(self, X):
        while X.ndim > PREFIX_DIMS+1:
            assert X.shape[-1] == X.shape[-2]
            X = X.diagonal(0, -1, -2)
        return X

    def forward(self, data):
        out = Data(self.enc_schema, batch_size=data.batch_size)
        for entity in self.schema.entities:
            for relation in self.schema.relations:
                if entity not in relation.entities:
                    continue
                else:
                    pooling_dims = self.get_pooling_dims(entity, relation)
                    data_rel = data[relation.id]
                    entity_out = self.pool_tensor(data_rel, pooling_dims)
                    entity_out = self.pool_tensor_diag(entity_out)
            out[entity.id] = entity_out
        return out

    

class SparseReLU(nn.Module):
    '''
    ReLU applied to sparse tensors
    '''
    def __init__(self):
        super(SparseReLU, self).__init__()

    def forward(self, sparse_tensor):
        return SparseTensor( sparse_tensor.indices, F.relu(sparse_tensor.values), sparse_tensor.shape)

class SparseMatrixReLU(nn.Module):
    '''
    ReLU applied to sparse matrix
    '''
    def __init__(self):
        super(SparseMatrixReLU, self).__init__()

    def forward(self, sparse_matrix):
        out =  sparse_matrix.clone()
        out.values = F.relu(out.values)
        return out