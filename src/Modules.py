#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 23:25:39 2021

@author: Daniel
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Modeul):
    def __init__(self, schema):
        super(LayerNorm, self).__init__()
        self.schema = schema
        self.layer_norm = nn.LayerNorm()
    
    def forward(self, X):
        for relation in self.schema.relations:
            X[relation.id] = (X[relation.id]) 
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

    def make_relation(self, encodings, relation):
        relation_shape = [self.dims] + relation.get_shape()
        relation_out = torch.zeros(relation_shape)
        num_new_dims = len(relation.entities) -1
        for entity_idx, entity in enumerate(relation.entities):
            entity_enc = encodings[entity.id]
            # Expand tensor to appropriate number of dimensions
            for _ in range(num_new_dims):
                entity_enc = entity_enc.unsqueeze(-1)
            # Transpose the entity to the appropriate dimension
            entity_enc.transpose_(1, entity_idx+1)
            # Broadcast-add to output
            relation_out += entity_enc
        return relation_out

    def forward(self, encodings):
        data_out = {}
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
    
    def get_pooling_dims(self, entity, relation):
        pooling_dims = []
        for entity_dim, rel_entity in enumerate(relation.entities):
            if entity != rel_entity:
                pooling_dims += [1 + entity_dim]
        return pooling_dims
    
    def pool_tensor(self, X, pooling_dims):
        if len(pooling_dims) > 0:
            return torch.sum(X, pooling_dims)
        else:
            return X
    
    def pool_tensor_diag(self, X):
        while X.ndim > 2:
            assert X.shape[-1] == X.shape[-2]
            X = X.diagonal(0, -1, -2)
        return X
    
    def forward(self, data):
        out = {}
        for entity in self.schema.entities:
            entity_out = torch.zeros(entity.n_instances)
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