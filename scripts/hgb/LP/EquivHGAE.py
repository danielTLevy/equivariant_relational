import torch
import torch.nn as nn
import torch.nn.functional as F
from src.DataSchema import DataSchema
from src.EquivariantLayer import EquivariantLayer
from src.SparseMatrixEquivariantLayer import SparseMatrixEquivariantLayer, \
                    SparseMatrixEntityPoolingLayer, SparseMatrixEntityBroadcastingLayer, SparseMatrixEquivariantSharingLayer
from src.SparseEquivariantLayer import SparseEquivariantLayer
from src.Modules import Activation,  Dropout,  SparseMatrixRelationLinear
import pdb

class DistMult(nn.Module):
    def __init__(self, num_rel, dim):
        super(DistMult, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(size=(num_rel, dim, dim)))
        nn.init.xavier_normal_(self.W, gain=1.414)

    def forward(self, left_emb, right_emb, r_id):
        thW = self.W[r_id].unsqueeze(0).repeat(len(left_emb), 1, 1)
        left_emb = torch.unsqueeze(left_emb, 1)
        right_emb = torch.unsqueeze(right_emb, 2)
        return torch.bmm(torch.bmm(left_emb, thW), right_emb).squeeze()

class Dot(nn.Module):
    def __init__(self):
        super(Dot, self).__init__()
    def forward(self, left_emb, right_emb, r_id):
        left_emb = torch.unsqueeze(left_emb, 1)
        right_emb = torch.unsqueeze(right_emb, 2)
        return torch.bmm(left_emb, right_emb).squeeze()

class EquivEncoder(nn.Module):
    '''
    Encoder to produce entity embeddings which can be used for
    node classification / regression or link prediction
    '''
    def __init__(self, schema, input_channels=1, activation=F.relu,
                 layers=[64, 64, 64], embedding_dim=50,
                 dropout=0, pool_op='mean', norm_affine=False,
                 embedding_entities = None,
                 in_fc_layer=True):
        super(EquivEncoder, self).__init__()
        self.schema = schema
        self.input_channels = input_channels

        self.activation = activation
        self.rel_activation = Activation(schema, self.activation, is_sparse=True)

        self.dropout = Dropout(p=dropout)
        self.rel_dropout  = Activation(schema, self.dropout, is_sparse=True)

        self.use_in_fc_layer = in_fc_layer
        # Equivariant Layeres
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
        self.norms = nn.ModuleList()
        for channels in layers:
            norm_dict = nn.ModuleDict()
            for rel_id in self.schema.relations:
                norm_dict[str(rel_id)] = nn.BatchNorm1d(channels, affine=norm_affine, track_running_stats=False)
            norm_activation = Activation(schema, norm_dict, is_dict=True, is_sparse=True)
            self.norms.append(norm_activation)

        # Entity embeddings
        self.pooling = SparseMatrixEntityPoolingLayer(schema, layers[-1],
                                                      embedding_dim,
                                                      entities=embedding_entities,
                                                      pool_op=pool_op)


    def forward(self, data, idx_identity=None, idx_transpose=None,
                data_embedding=None):
        if idx_identity is None or idx_transpose is None:
            print("Calculating idx_identity and idx_transpose. This can be precomputed.")
            idx_identity, idx_transpose = data.calculate_indices()
        if self.use_in_fc_layer:
            data = self.fc_in_layer(data)
        for i in range(self.n_equiv_layers):
            data = self.rel_dropout(self.rel_activation(self.norms[i](
                    self.equiv_layers[i](data, idx_identity, idx_transpose))))
        data = self.pooling(data, data_embedding)
        return data


class EquivDecoder(nn.Module):
    '''
    
    '''
    def __init__(self, schema, activation=F.relu,
                 layers=[64, 64, 64], embedding_dim=50,
                 dropout=0, pool_op='mean', norm_affine=False,
                 embedding_entities = None,
                 out_fc_layer=True,
                 out_dim=1):
        super(EquivDecoder, self).__init__()
        self.schema = schema
        self.out_dim = out_dim
        self.activation = activation
        self.rel_activation = Activation(schema, self.activation, is_sparse=True)

        self.dropout = Dropout(p=dropout)
        self.rel_dropout  = Activation(schema, self.dropout, is_sparse=True)

        self.use_out_fc_layer = out_fc_layer

        # Equivariant Layers
        self.broadcasting = SparseMatrixEntityBroadcastingLayer(self.schema,
                                                                embedding_dim,
                                                                layers[0],
                                                                entities=embedding_entities,
                                                                pool_op=pool_op)

        self.equiv_layers = nn.ModuleList([])
        self.equiv_layers.extend([
                SparseMatrixEquivariantLayer(schema, layers[i-1], layers[i], pool_op=pool_op)
                for i in range(1, len(layers))])
        self.n_layers = len(layers) - 1
        if self.use_out_fc_layer:
            # Add fully connected layer to output
            self.fc_out_layer = SparseMatrixRelationLinear(schema, layers[-1], self.out_dim)
        else:
            # Alternatively, use an equivariant layer
            self.equiv_layers.append(SparseMatrixEquivariantLayer(
                schema, layers[-1], self.out_dim, pool_op=pool_op))

        self.norms = nn.ModuleList()
        for channels in layers:
            norm_dict = nn.ModuleDict()
            for rel_id in self.schema.relations:
                norm_dict[str(rel_id)] = nn.BatchNorm1d(channels, affine=norm_affine, track_running_stats=False)
            norm_activation = Activation(schema, norm_dict, is_dict=True, is_sparse=True)
            self.norms.append(norm_activation)


    def forward(self, data_embedding, idx_identity=None, idx_transpose=None,
                data_target=None):
        if idx_identity is None or idx_transpose is None:
            print("Calculating idx_identity and idx_transpose. This can be precomputed.")
            idx_identity, idx_transpose = data_target.calculate_indices()

        data = self.broadcasting(data_embedding, data_target)
        for i in range(self.n_layers):
            data = self.rel_dropout(self.rel_activation(self.norms[i](
                    self.equiv_layers[i](data, idx_identity, idx_transpose))))
        if self.use_out_fc_layer:
            data = self.fc_out_layer(data)
        else:
            data = self.equiv_layers[-1](data, idx_identity, idx_transpose)
        return data


class EquivLinkPredictor(nn.Module):
    def __init__(self, schema, input_channels=1, activation=F.relu,
                 layers=[64, 64, 64], embedding_dim=50,
                 dropout=0,  pool_op='mean', norm_affine=False,
                 norm_embed=False,
                 final_activation=nn.Identity(),
                 embedding_entities = None,
                 output_rels = None,
                 in_fc_layer=True,
                 decode = 'dot',
                 out_dim=1):
        super(EquivLinkPredictor, self).__init__()
        self.output_rels = output_rels
        if output_rels == None:
            self.schema_out = schema
        else:
            self.schema_out = DataSchema(schema.entities, output_rels)
        self.out_dim = out_dim
        self.encoder = EquivEncoder(schema, input_channels, activation, layers,
                                    embedding_dim, dropout, pool_op,
                                    norm_affine, embedding_entities, in_fc_layer)
        self.norm_embed = norm_embed
        self.decode = decode
        if self.decode == 'dot':
            self.decoder = Dot()
        elif self.decode == 'distmult':
            self.decoder = DistMult(len(schema.relations), embedding_dim)
        elif decode == 'equiv':
            self.decoder = EquivDecoder(self.schema_out, activation,
                 layers, embedding_dim,
                 dropout, pool_op, norm_affine,
                 embedding_entities,
                 out_fc_layer=in_fc_layer,
                 out_dim=self.out_dim)
        elif self.decode == 'broadcast':
            self.decoder = SparseMatrixEntityBroadcastingLayer(self.schema_out,
                                                                embedding_dim,
                                                                input_channels,
                                                                entities=embedding_entities,
                                                                pool_op=pool_op)

        self.final_activation = Activation(self.schema_out, final_activation, is_sparse=True)

    def forward(self, data, idx_identity=None, idx_transpose=None,
                data_embedding=None, data_target=None,
                idx_id_out=None, idx_trans_out=None):
        embeddings = self.encoder(data, idx_identity, idx_transpose, data_embedding)
        if self.norm_embed:
            for entity_id, embedding in embeddings.items():
                embeddings[entity_id].values = F.normalize(embedding.values, 2, 1)
        if self.decode == 'dot' or self.decode == 'distmult':
            data_out = data_target.clone()
            for rel_id, output_rel in self.schema_out.relations.items():
                left_id = output_rel.entities[0].id
                left_target_indices = data_target[rel_id].indices[0]
                left = embeddings[left_id].values[left_target_indices]
                right_id = output_rel.entities[1].id
                right_target_indices = data_target[rel_id].indices[1]
                right = embeddings[right_id].values[right_target_indices]
                data_out[rel_id].values =  self.decoder(left, right, rel_id).unsqueeze(1)
            return data_out
        elif self.decode == 'broadcast':
            return self.decoder(embeddings, data_target)
        elif self.decode == 'equiv':
            return self.decoder(embeddings, idx_id_out, idx_trans_out, data_target)

class EquivAlternatingLinkPredictor(nn.Module):
    def __init__(self, schema, input_channels,
                 width, depth, embedding_dim,
                  activation=F.relu,
                 final_activation=nn.Identity(),
                 output_dim=1,  dropout=0, norm=True, pool_op='mean',
                 in_fc_layer=True, out_fc_layer=True,
                 norm_affine=False,
                 residual=False):
        super(EquivAlternatingLinkPredictor, self).__init__()

        self.schema = schema
        self.input_channels = input_channels

        self.width = width
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.activation = activation
        self.rel_activation = Activation(schema, self.activation, is_sparse=True)

        self.dropout = Dropout(p=dropout)
        self.rel_dropout  = Activation(schema, self.dropout, is_sparse=True)

        self.use_in_fc_layer = in_fc_layer
        if self.use_in_fc_layer:
            self.in_fc_layer = SparseMatrixRelationLinear(schema, self.input_channels,
                                                          width)
        self.use_out_fc_layer = out_fc_layer
        if self.use_out_fc_layer:
            self.out_fc_layer = SparseMatrixRelationLinear(schema, width,
                                                          output_dim)

        # Equivariant Layers
        self.pool_layers = nn.ModuleList([])
        self.bcast_layers = nn.ModuleList([])

        for i in range(depth):
            if i == 0 and not self.use_in_fc_layer:
                in_dim = input_channels
            else:
                in_dim = width
            if i == depth - 1 and not self.use_out_fc_layer:
                out_dim = output_dim
            else:
                out_dim = width

            pool_i = SparseMatrixEntityPoolingLayer(schema, in_dim,
                                                      embedding_dim,
                                                      entities=schema.entities,
                                                      pool_op=pool_op)
            self.pool_layers.append(pool_i)

            bcast_i = SparseMatrixEntityBroadcastingLayer(schema, embedding_dim,
                                                          out_dim,
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
        self.residual = residual

    def forward(self, data, indices_identity, indices_transpose, data_embedding, data_target):
        if self.use_in_fc_layer:
            data = self.rel_activation(self.in_fc_layer(data))
        data_prev = data.clone().zero_()
        data_embedding_prev = data_embedding.clone().zero_()
        for i in range(self.depth):
            data_embedding = self.rel_dropout(self.norms[i](self.rel_activation(self.pool_layers[i](data, data_embedding))))
            # Add residual
            if self.residual:
                data_embedding = data_embedding + data_embedding_prev
                data_embedding_prev = data_embedding

            if i == self.depth - 1:
                data = self.bcast_layers[i](data_embedding, data_target)
            else:
                data = self.rel_dropout(self.rel_activation(self.bcast_layers[i](data_embedding, data)))
            # Add residual
            if self.residual:
                data = data + data_prev
                data_prev = data
        if self.use_out_fc_layer:
            data = self.out_fc_layer(data)
        out = self.final_activation(data)
        return out

class EquivLinkPredictorShared(EquivLinkPredictor):
    def __init__(self, schema, input_channels=1, activation=F.relu,
                 layers=[64, 64, 64], embedding_dim=50,
                 dropout=0,  pool_op='mean', norm_affine=False,
                 norm_embed=False,
                 final_activation=nn.Identity(),
                 embedding_entities = None,
                 output_rels = None,
                 in_fc_layer=True,
                 decode = 'dot',
                 out_dim=1):
        super(EquivLinkPredictorShared, self).__init__(schema, input_channels, activation,
                     layers, embedding_dim,
                     dropout,  pool_op, norm_affine,
                     norm_embed,
                     final_activation,
                     embedding_entities,
                     output_rels,
                     in_fc_layer,
                     decode,
                     out_dim)
        if self.encoder.use_in_fc_layer:
            # All equiv layers have same dimension
            start_index = 0
        else:
            # First equiv layer is of different dimension
            start_index = 1
        for i in range(len(self.encoder.equiv_layers) - start_index):
            input_dim = layers[i]
            output_dim = layers[i + 1]
            self.encoder.equiv_layers[i + start_index] = SparseMatrixEquivariantSharingLayer(schema, input_dim, output_dim, pool_op=pool_op)

        if self.decoder.use_out_fc_layer:
            # Last equiv layer is different dimension
            n_decoder = len(self.decoder.equiv_layers)
        else:
            n_decoder = len(self.decoder.equiv_layers) - 1
        for i in range(n_decoder):
            input_dim = layers[i]
            output_dim = layers[i + 1]
            self.decoder.equiv_layers[i] = SparseMatrixEquivariantSharingLayer(schema, input_dim, output_dim, pool_op=pool_op)


class EquivLinkPredictorAblation(EquivLinkPredictor):
    '''
    Network for predicting properties of a single entity, where relations
    take the form of sparse matrices
    Can choose a set of parameters to turn off in all layers
    '''
    def __init__(self, schema, input_channels=1, activation=F.relu,
                 layers=[64, 64, 64], embedding_dim=50,
                 dropout=0,  pool_op='mean', norm_affine=False,
                 final_activation=nn.Identity(),
                 embedding_entities = None,
                 output_rels = None,
                 in_fc_layer=True,
                 decode = 'dot',
                 out_dim=1,
                 removed_params = None):

        super(EquivLinkPredictorAblation, self).__init__(schema, input_channels, activation,
                                                layers, embedding_dim,
                                                dropout, pool_op,
                                                norm_affine,
                                                final_activation,
                                                embedding_entities,
                                                output_rels,
                                                in_fc_layer,
                                                decode,out_dim)

        self.removed_params = [] if removed_params == None else removed_params
        self.remove_params(self.removed_params)

    def remove_params(self, params):
        for equiv_layer in self.encoder.equiv_layers:
            block = equiv_layer.block_modules[str((0,0))]
            for index in sorted(params, reverse=True):
                del block.all_ops[index]
            block.n_params = len(block.all_ops)

        if self.decode == 'equiv':
            for equiv_layer in self.decoder.equiv_layers:
                block = equiv_layer.block_modules[str((0,0))]
                for index in sorted(params, reverse=True):
                    del block.all_ops[index]
                block.n_params = len(block.all_ops)


class EquivHGAE(nn.Module):
    '''
    Autoencoder to produce entity embeddings which can be used for
    node classification / regression or link prediction
    '''
    def __init__(self, schema, input_channels=1, activation=F.relu,
                 layers=[64, 64, 64], embedding_dim=50,
                 dropout=0, norm=True, pool_op='mean', norm_affine=False,
                 final_activation=nn.Identity(),
                 embedding_entities = None,
                 output_relations = None,
                 in_fc_layer=True):
        super(EquivHGAE, self).__init__()
        self.schema = schema
        if output_relations == None:
            self.schema_out = schema
        else:
            self.schema_out = DataSchema(schema.entities,
                                         {rel.id: rel for rel in output_relations})
        self.input_channels = input_channels

        self.activation = activation
        self.rel_activation = Activation(schema, self.activation, is_sparse=True)

        self.dropout = Dropout(p=dropout)
        self.rel_dropout  = Activation(schema, self.dropout, is_sparse=True)

        self.use_in_fc_layer = in_fc_layer
        # Equivariant Layeres
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

        # Entity embeddings
        self.pooling = SparseMatrixEntityPoolingLayer(schema, layers[-1],
                                                      embedding_dim,
                                                      entities=embedding_entities,
                                                      pool_op=pool_op)
        self.broadcasting = SparseMatrixEntityBroadcastingLayer(self.schema_out,
                                                                embedding_dim,
                                                                input_channels,
                                                                entities=embedding_entities,
                                                                pool_op=pool_op)

        self.final_activation = Activation(schema, final_activation, is_sparse=True)



    def forward(self, data, idx_identity=None, idx_transpose=None,
                data_target=None, data_embedding=None, get_embeddings=False):
        if idx_identity is None or idx_transpose is None:
            print("Calculating idx_identity and idx_transpose. This can be precomputed.")
            idx_identity, idx_transpose = data.calculate_indices()
        if self.use_in_fc_layer:
            data = self.fc_in_layer(data)
        for i in range(self.n_equiv_layers):
            data = self.rel_dropout(self.rel_activation(self.norms[i](
                    self.equiv_layers[i](data, idx_identity, idx_transpose))))
        data = self.pooling(data, data_embedding)
        if get_embeddings:
            return data
        data = self.broadcasting(data, data_target)
        data = self.final_activation(data)
        return data
