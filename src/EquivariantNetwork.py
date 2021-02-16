# -*- coding: utf-8 -*-
from DataSchema import DataSchema, Entity, Relation
from EquivariantLayer import EquivariantLayer
from Modules import RelationNorm, ReLU, EntityPooling, EntityBroadcasting

import torch
import torch.nn as nn

        
class EquivariantNetwork(nn.Module):
    def __init__(self, data_schema):
        super(EquivariantNetwork, self).__init__()
        self.data_schema = data_schema
        self.ReLU = ReLU(data_schema)
        self.eerl0 = EquivariantLayer(data_schema, 1, 1)
        #self.eerl1 = EquivariantLayer(data_schema, 8, 16)
        #self.eerl2 = EquivariantLayer(data_schema, 16, 16)
        #self.eerl3 = EquivariantLayer(data_schema, 8, 1)
        #self.eerl4 = EquivariantLayer(data_schema)

    def forward(self, data):
            
        #out = forward_pass(data, self.eerl0)
        out = self.eerl0(data)
        #out = forward_pass(out, self.eerl1)
        #out = forward_pass(out, self.eerl2)
        #out = self.eerl3(out)

        return out


class EquivariantAutoEncoder(nn.Module):
    def __init__(self, data_schema, hidden_dim = 10):
        super(EquivariantAutoEncoder, self).__init__()
        self.data_schema = data_schema
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
                EquivariantLayer(data_schema, 1, 16),
                ReLU(data_schema),
                RelationNorm(data_schema, 16, affine=True),

                EquivariantLayer(data_schema, 16, self.hidden_dim),
                RelationNorm(data_schema, self.hidden_dim, affine=True),

                EntityPooling(data_schema, self.hidden_dim)
                )
        self.decoder = nn.Sequential(
                EntityBroadcasting(data_schema, self.hidden_dim),
                EquivariantLayer(data_schema, self.hidden_dim, 16),
                ReLU(data_schema),
                RelationNorm(data_schema, 16, affine=True),

                EquivariantLayer(data_schema, 16, 1),
                RelationNorm(data_schema, 1, affine=False)
                )


    def get_encoding_size(self):
        return {entity.id: (entity.n_instances, self.hidden_dim) 
                for entity in self.data_schema.entities}
        
    def forward(self, data):
        enc  = self.encoder(data)
        out = self.decoder(enc)
        return out

