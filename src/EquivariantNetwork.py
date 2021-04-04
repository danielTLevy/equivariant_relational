# -*- coding: utf-8 -*-
import torch.nn as nn
from src.EquivariantLayer import EquivariantLayer
from src.SparseEquivariantLayer import SparseEquivariantLayer
from src.Modules import RelationNorm, Activation, EntityPooling, EntityBroadcasting, SparseReLU


class EquivariantNetwork(nn.Module):
    def __init__(self, data_schema):
        super(EquivariantNetwork, self).__init__()
        self.data_schema = data_schema
        self.hidden_dims = (32, 64, 32)
        self.all_dims = [1] + list(self.hidden_dims) + [1]
        
        self.ReLU = Activation(data_schema, )
        sequential = []
        for i in range(1, len(self.all_dims)-1):
            sequential.append(EquivariantLayer(self.data_schema, self.all_dims[i-1], self.all_dims[i]))
            sequential.append(self.ReLU)
            sequential.append(RelationNorm(self.data_schema, self.all_dims[i], affine=False))
        sequential.append(EquivariantLayer(self.data_schema, self.all_dims[-2], self.all_dims[-1]))
        self.sequential = nn.Sequential(*sequential)
        
    def forward(self, data):
        out = self.sequential(data)
        return out


class EquivariantAutoEncoder(nn.Module):
    def __init__(self, data_schema, encoding_dim = 10):
        super(EquivariantAutoEncoder, self).__init__()
        self.data_schema = data_schema
        self.encoding_dim = encoding_dim
        self.dropout_rate = 0.2
        self.hidden_dims = [64]*2
        self.all_dims = [1] + list(self.hidden_dims) + [self.encoding_dim]
        self.n_layers = len(self.all_dims)

        self.ReLU = Activation(data_schema, nn.ReLU())
        self.Dropout = Activation(data_schema, nn.Dropout(p=self.dropout_rate))

        encoder = []
        encoder.append(EquivariantLayer(self.data_schema, self.all_dims[0], self.all_dims[1]))
        for i in range(1, self.n_layers-1):
            encoder.append(self.ReLU)
            encoder.append(RelationNorm(self.data_schema, self.all_dims[i], affine=False))
            encoder.append(self.Dropout)
            encoder.append(EquivariantLayer(self.data_schema, self.all_dims[i], self.all_dims[i+1]))
        encoder.append(EntityPooling(data_schema, self.encoding_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        decoder.append(EntityBroadcasting(data_schema, self.encoding_dim))
        decoder.append(EquivariantLayer(self.data_schema, self.all_dims[-1], self.all_dims[-2]))
        for i in range(self.n_layers-2, 0, -1):
            decoder.append(self.ReLU)
            decoder.append(RelationNorm(self.data_schema, self.all_dims[i], affine=False))
            decoder.append(self.Dropout)
            decoder.append(EquivariantLayer(self.data_schema, self.all_dims[i], self.all_dims[i-1]))
        self.decoder = nn.Sequential(*decoder)

    def get_encoding_size(self):
        return {entity.id: (entity.n_instances, self.hidden_dim) 
                for entity in self.data_schema.entities}

    def forward(self, data):
        enc  = self.encoder(data)
        out = self.decoder(enc)
        return out

class SparseEquivariantNetwork(nn.Module):
    def __init__(self, data_schema, n_channels):
        super(SparseEquivariantNetwork, self).__init__()
        self.data_schema = data_schema
        self.n_channels = n_channels
        self.hidden_dims = (32, 64, 32)
        self.all_dims = [n_channels] + list(self.hidden_dims) + [n_channels]

        self.ReLU = Activation(data_schema, SparseReLU())
        sequential = []
        for i in range(1, len(self.all_dims)-1):
            sequential.append(SparseEquivariantLayer(self.data_schema, self.all_dims[i-1], self.all_dims[i]))
            sequential.append(self.ReLU)
            #sequential.append(RelationNorm(self.data_schema, self.all_dims[i], affine=False))
        sequential.append(SparseEquivariantLayer(self.data_schema, self.all_dims[-2], self.all_dims[-1]))
        self.sequential = nn.Sequential(*sequential)

    def forward(self, data):
        out = self.sequential(data)
        return out
