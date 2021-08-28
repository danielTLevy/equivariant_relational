import torch.nn as nn
import torch.nn.functional as F
from src.DataSchema import DataSchema
from src.EquivariantLayer import EquivariantLayer
from src.SparseMatrixEquivariantLayer import SparseMatrixEquivariantLayer, \
                    SparseMatrixEntityPoolingLayer, SparseMatrixEntityBroadcastingLayer
from src.SparseEquivariantLayer import SparseEquivariantLayer
from src.Modules import Activation,  Dropout,  SparseMatrixRelationLinear


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
                 output_relations = None):
        super(EquivHGAE, self).__init__()
        self.schema = schema
        if output_relations == None:
            self.schema_out = schema
        else:
            self.schema_out = DataSchema(schema.entities, output_relations)
        self.input_channels = input_channels

        self.activation = activation
        self.rel_activation = Activation(schema, self.activation, is_sparse=True)

        self.dropout = Dropout(p=dropout)
        self.rel_dropout  = Activation(schema, self.dropout, is_sparse=True)

        self.n_equiv_layers = len(layers)
        self.equiv_layers = nn.ModuleList([])
        self.equiv_layers.append(SparseMatrixEquivariantLayer(
                schema, input_channels, layers[0], pool_op=pool_op))
        self.equiv_layers.extend([
                SparseMatrixEquivariantLayer(schema, layers[i-1], layers[i], pool_op=pool_op)
                for i in range(1, len(layers))])
        if norm:
            self.norms = nn.ModuleList()
            for channels in layers:
                norm_dict = nn.ModuleDict()
                for relation in self.schema.relations:
                    norm_dict[str(relation.id)] = nn.BatchNorm1d(channels, affine=norm_affine, track_running_stats=False)
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
        for i in range(self.n_equiv_layers):
            data = self.rel_dropout(self.rel_activation(self.norms[i](
                    self.equiv_layers[i](data, idx_identity, idx_transpose))))
        data = self.pooling(data, data_embedding)
        if get_embeddings:
            return data
        data = self.broadcasting(data, data_target)
        data = self.final_activation(data)
        return data
