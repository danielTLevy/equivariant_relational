# -*- coding: utf-8 -*-
from DataSchema import DataSchema, Entity, Relation
from EquivariantLayer import EquivariantLayer
from Modules import ReLU, EntityPooling, EntityBroadcasting

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
    def __init__(self, data_schema):
        super(EquivariantAutoEncoder, self).__init__()
        self.hidden_dim = 5
        
        self.data_schema = data_schema
        self.ReLU = ReLU(data_schema)
        self.eerl1 = EquivariantLayer(data_schema, 1, 16)
        self.eerl2 = EquivariantLayer(data_schema, 16, self.hidden_dim)
        self.pooling = EntityPooling(data_schema, self.hidden_dim)
        self.broadcasting = EntityBroadcasting(data_schema, self.hidden_dim)
        self.eerl3 = EquivariantLayer(data_schema, self.hidden_dim, 16)
        self.eerl4 = EquivariantLayer(data_schema, 16, 1)

    def get_encoding_size(self):
        return {entity.id: (entity.n_instances, self.hidden_dim) 
                for entity in self.schema.entities}
        
    def forward(self, data):
        out = self.ReLU(self.eerl1(data))
        out = self.ReLU(self.eerl2(data))
        enc = self.pooling(out)
        
        out = self.broadcasting(enc)
        out = self.ReLU(self.eerl3(out))
        out = self.eerl4(out)
        return out

