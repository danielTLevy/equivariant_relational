# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.DataSchema import DataSchema
from src.EquivariantLayer import EquivariantLayer
from src.SparseMatrixEquivariantLayer import SparseMatrixEquivariantLayer, \
                    SparseMatrixEntityPoolingLayer, SparseMatrixEntityBroadcastingLayer, \
                    SparseMatrixEquivariantSharingLayer
from src.SparseEquivariantLayer import SparseEquivariantLayer
from src.Modules import Activation,  Dropout,  SparseMatrixRelationLinear
from src.utils import MATRIX_OPS

class EquivHGNet(nn.Module):
    '''
    Network for predicting properties of a single entity, where relations
    take the form of sparse matrices
    '''
    def __init__(self, schema, input_channels, activation=F.relu, 
                 layers=[32, 64, 32], target_entities=None,
                 fc_layers=[], final_activation=nn.Identity(),
                 output_dim=1,  dropout=0, norm=True, pool_op='mean',
                 in_fc_layer=True, mid_fc_layer=False,
                 norm_affine=False, norm_out=False,
                 residual=False):
        super(EquivHGNet, self).__init__()

        self.schema = schema
        self.input_channels = input_channels

        self.activation = activation
        self.rel_activation = Activation(schema, self.activation, is_sparse=True)

        self.dropout = Dropout(p=dropout)
        self.rel_dropout  = Activation(schema, self.dropout, is_sparse=True)

        self.use_in_fc_layer = in_fc_layer
        # Equivariant Layers
        self.equiv_layers = nn.ModuleList([])
        if self.use_in_fc_layer:
            # Simple fully connected layers for input attributes
            self.fc_in_layer = SparseMatrixRelationLinear(schema, self.input_channels,
                                                          layers[0])
            self.n_equiv_layers = len(layers) - 1
        else:
            # Alternatively, use an equivariant layer
            self.equiv_layers.append(SparseMatrixEquivariantLayer(
                schema, input_channels, layers[0], pool_op=pool_op))
            self.n_equiv_layers = len(layers)

        self.equiv_layers.extend([
                SparseMatrixEquivariantLayer(schema, layers[i-1], layers[i], pool_op=pool_op)
                for i in range(1, len(layers))])
        if norm:
            self.norms = nn.ModuleList()
            for channels in layers:
                norm_dict = nn.ModuleDict()
                for rel_id in self.schema.relations:
                    norm_dict[str(rel_id)] = nn.BatchNorm1d(channels, affine=norm_affine, track_running_stats=False)
                norm_activation = Activation(schema, norm_dict, is_dict=True, is_sparse=True)
                self.norms.append(norm_activation)
        else:
            self.norms = nn.ModuleList([Activation(schema, nn.Identity(), is_sparse=True)
                                        for _ in layers])
        # Optionally add intermediary fully connected layers for each 
        # equivariant layer
        self.use_mid_fc_layer = mid_fc_layer
        if self.use_mid_fc_layer:
            equiv_start_i = 1 if self.use_in_fc_layer else 0
            self.fc_mid_layers = nn.ModuleList([
                SparseMatrixRelationLinear(schema, layers[i], layers[i])
                for i in range(equiv_start_i, len(layers))])

        # Entity embeddings
        embedding_layers = fc_layers + [output_dim]
        self.pooling = SparseMatrixEntityPoolingLayer(schema, layers[-1],
                                                      embedding_layers[0],
                                                      entities=target_entities,
                                                      pool_op=pool_op)
        self.n_fc_layers = len(fc_layers)
        self.fc_layers = nn.ModuleList([])
        self.fc_layers.extend([nn.Linear(embedding_layers[i-1], embedding_layers[i])
                            for i in range(1, self.n_fc_layers+1)])
        self.final_activation = final_activation
        self.norm_out = norm_out
        self.residual = residual

    def forward(self, data, idx_identity=None, idx_transpose=None, data_out=None, get_embeddings=False):
        if idx_identity is None or idx_transpose is None:
            print("Calculating idx_identity and idx_transpose. This can be precomputed.")
            idx_identity, idx_transpose = data.calculate_indices()
        if self.use_in_fc_layer:
            data = self.fc_in_layer(data)
        for i in range(self.n_equiv_layers):
            equiv_out = self.equiv_layers[i](data, idx_identity, idx_transpose)
            if self.residual and i != 0:
                equiv_out = equiv_out + data
            data = self.rel_dropout(self.rel_activation(self.norms[i](
                    equiv_out)))
            if self.use_mid_fc_layer:
                data = self.rel_activation(self.fc_mid_layers[i](data))
        data = self.pooling(data, data_out)
        out = data[0].values
        if self.n_fc_layers > 0 and get_embeddings == False:
            out = self.fc_layers[0](out)
            for i in range(1, self.n_fc_layers):
                out = self.fc_layers[i](self.dropout(self.activation(out)))
        if self.norm_out:
            out = F.normalize(out, p=2., dim=1)
        out = self.final_activation(out)
        return out

class EquivHGNetShared(EquivHGNet):
    '''
    Network for predicting properties of a single entity, where relations
    take the form of sparse matrices
    Unlike regular EquivHGNEt, layers (other than first and last) can include
    parameters shared across relation pairs
    '''
    def __init__(self, schema, input_channels, activation=F.relu, 
                 layers=[32, 64, 32], target_entities=None,
                 fc_layers=[], final_activation=nn.Identity(),
                 output_dim=1,  dropout=0, norm=True, pool_op='mean',
                 in_fc_layer=True, mid_fc_layer=False,
                 norm_affine=False, norm_out=False,
                 residual=False):
        super(EquivHGNetShared, self).__init__(schema, input_channels, activation, 
                     layers, target_entities,
                     fc_layers, final_activation,
                     output_dim,  dropout, norm, pool_op,
                     in_fc_layer, mid_fc_layer,
                     norm_affine, norm_out,
                     residual)

        # Convert SparseMatrixEquivariantLayer to SparseMatrixEquivariantSharingLayer
        # If not using an FC layer at input, then input relations will have
        # different dimensions and thus can't be shared
        if self.use_in_fc_layer:
            start_index = 0
        else:
            start_index = 1
        for i in range(start_index, len(self.equiv_layers)):
            input_dim = layers[i - start_index]
            output_dim = layers[i - start_index + 1]
            self.equiv_layers[i] = SparseMatrixEquivariantSharingLayer(schema, input_dim, output_dim, pool_op=pool_op)



class EquivHGNetAblation(EquivHGNet):
    '''
    Network for predicting properties of a single entity, where relations
    take the form of sparse matrices
    Can choose a set of parameters to turn off in all layers
    '''
    def __init__(self, schema, input_channels, activation=F.relu, 
                 layers=[32, 64, 32], target_entities=None,
                 fc_layers=[], final_activation=nn.Identity(),
                 output_dim=1,  dropout=0, norm=True, pool_op='mean',
                 in_fc_layer=True, mid_fc_layer=False,
                 norm_affine=False, norm_out=False,
                 residual=False,
                 removed_params = None):
        super(EquivHGNetAblation, self).__init__(schema, input_channels, activation,
                                                 layers, target_entities,
                                                 fc_layers, final_activation,
                                                 output_dim, dropout,
                                                 norm, pool_op,
                                                 in_fc_layer, mid_fc_layer,
                                                 norm_affine, norm_out,
                                                 residual)

        self.removed_params = [] if removed_params == None else removed_params
        self.remove_params(self.removed_params)

    def remove_params(self, params):
        '''
        Given these 15 "canonical" operations, get corresponding indices for
        the same ops in each individual relation and remove them
        '''

        for equiv_layer in self.equiv_layers:
            for block_id, block in equiv_layer.block_modules.items():
                block_params = []
                for index in sorted(params, reverse=True):
                    op_name = MATRIX_OPS[index]
                    if op_name in block.all_ops:
                        block_params.append(block.all_ops.index(op_name))
                for index in sorted(block_params, reverse=True):
                    del block.all_ops[index]
                block.n_params -= len(block_params)
            block.n_params = len(block.all_ops)



class AlternatingHGN(nn.Module):
    '''
    Network for predicting properties of a single entity, where relations
    take the form of sparse matrices.
    Alternates between second-order and first-order embeddings using pooling
    and broadcasting operations
    '''
    def __init__(self, schema, input_channels, 
                 width, depth, embedding_dim,
                  activation=F.relu,
                 fc_layers=[], final_activation=nn.Identity(),
                 output_dim=1,  dropout=0, norm=True, pool_op='mean',
                 in_fc_layer=True,
                 norm_affine=False,
                 norm_out=False):
        super(AlternatingHGN, self).__init__()

        self.schema = schema
        self.input_channels = input_channels

        self.width = width
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.activation = activation
        self.rel_activation = Activation(schema, self.activation, is_sparse=True)

        self.dropout = Dropout(p=dropout)
        #self.rel_dropout  = Activation(schema, self.dropout, is_sparse=True)

        self.use_in_fc_layer = in_fc_layer
        # Equivariant Layers
        self.pool_layers = nn.ModuleList([])
        self.bcast_layers = nn.ModuleList([])

        for i in range(depth + 1):
            if i == 0:
                in_dim = input_channels
            else:
                in_dim = width
            if i == depth:
                out_dim = output_dim
            else:
                out_dim = embedding_dim

            pool_i = SparseMatrixEntityPoolingLayer(schema, in_dim,
                                                      out_dim,
                                                      entities=schema.entities,
                                                      pool_op=pool_op)
            self.pool_layers.append(pool_i)

        for i in range(depth):
            bcast_i = SparseMatrixEntityBroadcastingLayer(schema, embedding_dim,
                                                          width,
                                                          entities=schema.entities,
                                                          pool_op=pool_op)
            self.bcast_layers.append(bcast_i)

        if norm:
            self.norms = nn.ModuleList()
            for i in range(depth):
                norm_dict = nn.ModuleDict()
                for rel_id in self.schema.relations:
                    norm_dict[str(rel_id)] = nn.BatchNorm1d(embedding_dim,
                                                            affine=norm_affine, track_running_stats=False)
                norm_activation = Activation(schema, norm_dict, is_dict=True, is_sparse=True)
                self.norms.append(norm_activation)
        else:
            self.norms = nn.ModuleList([Activation(schema, nn.Identity(), is_sparse=True)
                                        for _ in range(depth)])


        self.final_activation = final_activation
        self.norm_out = norm_out

    def forward(self, data, data_embedding):
        for i in range(self.depth):
            data_embedding = self.norms[i](self.rel_activation(self.pool_layers[i](data, data_embedding)))

            data = self.rel_activation(self.bcast_layers[i](data_embedding, data))

        out = self.pool_layers[-1](data, data_embedding)[0].values
        if self.norm_out:
            out = F.normalize(out, p=2., dim=1)
        out = self.final_activation(out)
        return out


