# -*- coding: utf-8 -*-
import torch.nn as nn

from src.EquivariantLayer import EquivariantLayer
from src.SparseMatrixEquivariantLayer import SparseMatrixEquivariantLayer
from src.SparseEquivariantLayer import SparseEquivariantLayer
from src.Modules import RelationNorm, Activation, EntityPooling, \
                        EntityBroadcasting, SparseActivation


class EquivariantNetwork(nn.Module):
    def __init__(self, data_schema, n_channels):
        super(EquivariantNetwork, self).__init__()
        self.data_schema = data_schema
        self.n_channels = n_channels
        self.hidden_dims = (32, 64, 32)
        self.all_dims = [n_channels] + list(self.hidden_dims) + [n_channels]
        
        self.ReLU = Activation(data_schema, nn.ReLU())
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

        self.ReLU = SparseActivation(data_schema, nn.ReLU())
        sequential = []
        for i in range(1, len(self.all_dims)-1):
            sequential.append(SparseEquivariantLayer(self.data_schema, self.all_dims[i-1], self.all_dims[i]))
            sequential.append(self.ReLU)
            sequential.append(RelationNorm(self.data_schema, self.all_dims[i], affine=False, sparse=True))
        sequential.append(SparseEquivariantLayer(self.data_schema, self.all_dims[-2], self.all_dims[-1]))
        self.sequential = nn.Sequential(*sequential)

    def forward(self, data):
        out = self.sequential(data)
        return out


class SparseMatrixEquivariantNetwork(nn.Module):
    def __init__(self, data_schema, n_channels, target_rel=None):
        super(SparseMatrixEquivariantNetwork, self).__init__()
        self.data_schema = data_schema
        self.n_channels = n_channels


        self.ReLU = SparseActivation(data_schema, nn.ReLU())
        self.layer1 = SparseMatrixEquivariantLayer(self.data_schema, n_channels, 32)
        self.norm1 = RelationNorm(self.data_schema, 32, affine=False,
                                  sparse=True, matrix=True)
        self.layer2 = SparseMatrixEquivariantLayer(self.data_schema, 32, 64)
        self.norm2 = RelationNorm(self.data_schema, 64, affine=False,
                                  sparse=True, matrix=True)
        self.layer3 = SparseMatrixEquivariantLayer(self.data_schema, 64, 32)
        self.norm3 = RelationNorm(self.data_schema, 32, affine=False,
                                  sparse=True, matrix=True)
        self.layer4 = SparseMatrixEquivariantLayer(self.data_schema, 32, n_channels,
                                                   target_rel=target_rel)
        #self.sigmoid = SparseActivation(data_schema, nn.Sigmoid())
        self.softmax = nn.Softmax()

    def forward(self, data, idx_identity=None, idx_transpose=None):
        if idx_identity is None or idx_transpose is None:
            print("Calculating idx_identity and idx_transpose. This can be precomputed.")
            idx_identity, idx_transpose = data.calculate_indices()
        out = self.norm1(self.ReLU(self.layer1(data, idx_identity, idx_transpose)))
        out = self.norm2(self.ReLU(self.layer2(out, idx_identity, idx_transpose)))
        out = self.norm3(self.ReLU(self.layer3(out, idx_identity, idx_transpose)))
        out = self.layer4(out, idx_identity, idx_transpose)
        data_out =  out[0].values.view(7, 2708).T
        data_out = self.softmax(data_out)
        return data_out