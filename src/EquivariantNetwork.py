# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from src.EquivariantLayer import EquivariantLayer
from src.SparseMatrixEquivariantLayer import SparseMatrixEquivariantLayer, SparseMatrixEntityPoolingLayer
from src.SparseEquivariantLayer import SparseEquivariantLayer
from src.Modules import RelationNorm, Activation, EntityPooling, \
                        EntityBroadcasting, SparseActivation, Dropout
import pdb

class EquivariantNetwork(nn.Module):
    def __init__(self, schema, n_channels):
        super(EquivariantNetwork, self).__init__()
        self.schema = schema
        self.n_channels = n_channels
        self.hidden_dims = (32, 64, 32)
        self.all_dims = [n_channels] + list(self.hidden_dims) + [n_channels]
        
        self.ReLU = Activation(schema, nn.ReLU())
        sequential = []
        for i in range(1, len(self.all_dims)-1):
            sequential.append(EquivariantLayer(self.schema, self.all_dims[i-1], self.all_dims[i]))
            sequential.append(self.ReLU)
            sequential.append(RelationNorm(self.schema, self.all_dims[i], affine=False))
        sequential.append(EquivariantLayer(self.schema, self.all_dims[-2], self.all_dims[-1]))
        self.sequential = nn.Sequential(*sequential)
        
    def forward(self, data):
        out = self.sequential(data)
        return out


class EquivariantAutoEncoder(nn.Module):
    def __init__(self, schema, encoding_dim = 10):
        super(EquivariantAutoEncoder, self).__init__()
        self.schema = schema
        self.encoding_dim = encoding_dim
        self.dropout_rate = 0.2
        self.hidden_dims = [64]*2
        self.all_dims = [1] + list(self.hidden_dims) + [self.encoding_dim]
        self.n_layers = len(self.all_dims)

        self.ReLU = Activation(schema, nn.ReLU())
        self.Dropout = Activation(schema, nn.Dropout(p=self.dropout_rate))

        encoder = []
        encoder.append(EquivariantLayer(self.schema, self.all_dims[0], self.all_dims[1]))
        for i in range(1, self.n_layers-1):
            encoder.append(self.ReLU)
            encoder.append(RelationNorm(self.schema, self.all_dims[i], affine=False))
            encoder.append(self.Dropout)
            encoder.append(EquivariantLayer(self.schema, self.all_dims[i], self.all_dims[i+1]))
        encoder.append(EntityPooling(schema, self.encoding_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        decoder.append(EntityBroadcasting(schema, self.encoding_dim))
        decoder.append(EquivariantLayer(self.schema, self.all_dims[-1], self.all_dims[-2]))
        for i in range(self.n_layers-2, 0, -1):
            decoder.append(self.ReLU)
            decoder.append(RelationNorm(self.schema, self.all_dims[i], affine=False))
            decoder.append(self.Dropout)
            decoder.append(EquivariantLayer(self.schema, self.all_dims[i], self.all_dims[i-1]))
        self.decoder = nn.Sequential(*decoder)

    def get_encoding_size(self):
        return {entity.id: (entity.n_instances, self.hidden_dim) 
                for entity in self.schema.entities}

    def forward(self, data):
        enc  = self.encoder(data)
        out = self.decoder(enc)
        return out

class SparseEquivariantNetwork(nn.Module):
    def __init__(self, schema, n_channels, target_rel=None, final_activation=None):
        super(SparseEquivariantNetwork, self).__init__()
        self.schema = schema
        self.n_channels = n_channels
        self.hidden_dims = (32, 64, 32)
        self.all_dims = [n_channels] + list(self.hidden_dims) + [n_channels]
        self.target_rel = target_rel
        self.ReLU = SparseActivation(schema, nn.ReLU())
        sequential = []
        for i in range(1, len(self.all_dims)-1):
            sequential.append(SparseEquivariantLayer(self.schema, self.all_dims[i-1], self.all_dims[i]))
            sequential.append(self.ReLU)
            sequential.append(RelationNorm(self.schema, self.all_dims[i], affine=False, sparse=True))
        sequential.append(SparseEquivariantLayer(self.schema, self.all_dims[-2],
                                                 self.all_dims[-1], target_rel=target_rel))

        self.sequential = nn.Sequential(*sequential)

        if final_activation == None:
            final_activation = nn.Identity()
        self.final = SparseActivation(schema, final_activation)

    def forward(self, data):
        out = self.sequential(data)
        if self.classification:
            out = out[self.target_rel].to_dense()[0,:]
        out = self.final(out)
        return out


class SparseMatrixEquivariantNetwork(nn.Module):
    def __init__(self, schema, input_channels, activation=F.relu, 
                 layers=[32, 64, 32], target_entities=None,
                 fc_layers=[], final_activation=nn.Identity(),
                 output_dim=1,  dropout=0):
        super(SparseMatrixEquivariantNetwork, self).__init__()
        self.schema = schema
        self.input_channels = input_channels

        self.activation = SparseActivation(schema, activation)
        self.dropout  = SparseActivation(schema, Dropout(p=dropout))

        self.n_equiv_layers = len(layers)
        self.equiv_layers = nn.ModuleList([])
        self.equiv_layers.append(SparseMatrixEquivariantLayer(
                schema, input_channels, layers[0]))
        self.equiv_layers.extend([
                SparseMatrixEquivariantLayer(schema, layers[i-1], layers[i])
                for i in range(1, len(layers))])
        self.norms = nn.ModuleList([RelationNorm(schema, channels, affine=False,
                                                 sparse=True, matrix=True)
                                    for channels in layers])
        embedding_layers = fc_layers + [output_dim]

        self.n_fc_layers = len(fc_layers)
        self.pooling = SparseMatrixEntityPoolingLayer(schema, layers[-1],
                                                      embedding_layers[0],
                                                      entities=target_entities)

        self.fc = nn.ModuleList([])
        self.fc.extend([nn.Linear(embedding_layers[i-1], embedding_layers[i])
                            for i in range(1, self.n_fc_layers+1)])

        self.final_activation = final_activation


    def forward(self, data, idx_identity=None, idx_transpose=None, data_out=None):
        if idx_identity is None or idx_transpose is None:
            print("Calculating idx_identity and idx_transpose. This can be precomputed.")
            idx_identity, idx_transpose = data.calculate_indices()
        for i in range(self.n_equiv_layers):
            data = self.norms[i](self.activation(self.dropout(
                    self.equiv_layers[i](data, idx_identity, idx_transpose))))
            #data = self.dropout(self.activation(self.norms[i](
            #        self.equiv_layers[i](data, idx_identity, idx_transpose))))
        data = self.dropout(self.pooling(data, data_out))
        out = data[0].values
        if self.n_fc_layers > 0:
            out = self.fc_layers[0](out)
            for i in range(1, self.n_fc_layers):
                out = self.fc_layers[i](self.dropout(self.activation(out)))
        out = self.final_activation(out)
        return out