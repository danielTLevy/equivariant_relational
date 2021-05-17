#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import torch
import torch.nn as nn
import itertools
import logging
from src.utils import get_all_ops, get_all_input_output_partitions, \
                        get_ops_from_partitions, MATRIX_PREFIX_DIMS
from src.DataSchema import SparseMatrixData, DataSchema, Data, Relation
from src.SparseMatrix import SparseMatrix
import pdb

LOG_LEVEL = logging.INFO

class SparseMatrixEquivariantLayerBlock(nn.Module):
    # Layer mapping between two relations
    def __init__(self, input_dim, output_dim, relation_in, relation_out):
        super(SparseMatrixEquivariantLayerBlock, self).__init__()
        assert len(relation_in.entities) == 2, "Relation must be second or first order"
        assert len(relation_out.entities) <= 2, "Relation must be second order"
        in_is_set = relation_in.is_set
        out_is_set = relation_out.is_set
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.input_output_partitions = get_all_input_output_partitions(relation_in, relation_out, MATRIX_PREFIX_DIMS)
        self.all_ops = get_ops_from_partitions(self.input_output_partitions,
                                               in_is_set, out_is_set)
        self.n_params = len(self.all_ops)
        stdv = 0.1 / math.sqrt(self.in_dim)
        self.weights = nn.Parameter(torch.Tensor(self.n_params, self.in_dim, self.out_dim).uniform_(-stdv, stdv))
        self.bias = nn.Parameter(torch.zeros(self.n_params))
        self.output_shape = [0, self.out_dim] + [entity.n_instances for entity in relation_out.entities]
        self.logger = logging.getLogger()
        self.logger.setLevel(LOG_LEVEL)

    def output_op(self, op_str, X_out, data, device):
        op, index_str = op_str.split('_')
        if op == 'b':
            return X_out.broadcast(data, index_str, device)
        elif op == 'e':
            return X_out.embed_diag(data, device)

    def input_op(self, op_str, X_in, device):
        op, index_str = op_str.split('_')
        if op == 'g':
            return X_in.gather_diag(device)
        elif op == 'p':
            return X_in.pool(index_str, device=device)

    def forward(self, X_in, X_out, indices_identity, indices_trans):
        '''
        X_in: Source sparse tensor
        X_out: Correpsonding sparse tensor for target relation
        '''
        self.logger.info("n_params: {}".format(self.n_params))
        if type(X_out) == SparseMatrix:
            Y = SparseMatrix.from_other_sparse_matrix(X_out, self.out_dim)
        else:
            Y = X_out.clone()
        #TODO: can add a cache for input operations here
        for i in range(self.n_params):
            op_inp, op_out = self.all_ops[i]
            weight = self.weights[i]
            device = weight.device
            if op_inp == None:
                X_mul = torch.matmul(X_in, weight)
                X_op_out = self.output_op(op_out, X_out, X_mul, device)
            elif op_out == None:
                X_op_inp = self.input_op(op_inp, X_in, device)
                X_mul = torch.matmul(X_op_inp, weight)
                X_op_out = X_mul
            elif op_out[0] == "i":
                # Identity
                X_intersection_vals = X_in.gather_mask(indices_identity[0])
                X_mul = X_intersection_vals @ weight
                X_op_out = X_out.broadcast_from_mask(X_mul, indices_identity[1], device)
            elif op_out[0] == "t":
                # Transpose
                X_T_intersection_vals = X_in.gather_transpose(indices_trans[0])
                X_mul = X_T_intersection_vals @ weight
                X_op_out = X_out.broadcast_from_mask(X_mul, indices_trans[1], device)
            else:
                # Pool or Gather or Do Nothing
                X_op_inp = self.input_op(op_inp, X_in, device)
                # Multiply values by weight
                X_mul = torch.matmul(X_op_inp, weight)
                # Broadcast or Embed Diag or Transpose
                X_op_out = self.output_op(op_out, X_out, X_mul, device)
            #assert X_op_out.nnz() == X_out.nnz()
            #assert Y.nnz() == X_out.nnz(), "Y: {}, X_out: {}".format(Y.nnz(), X_out.nnz())
            #assert Y.nnz() == X_op_out.nnz(), "Y: {}, X_op_out: {}".format(Y.nnz(), X_op_out.nnz())
            Y = Y + X_op_out + self.bias[i]
        return Y


class SparseMatrixEquivariantLayer(nn.Module):
    def __init__(self, schema, input_dim=1, output_dim=1, schema_out=None):
        '''
        input_dim: either a rel_id: dimension dict, or an integer for all relations
        output_dim: either a rel_id: dimension dict, or an integer for all relations
        '''
        super(SparseMatrixEquivariantLayer, self).__init__()
        self.schema = schema
        self.schema_out = schema_out
        if self.schema_out == None:
            self.schema_out = schema
        self.relation_pairs = list(itertools.product(self.schema.relations,
                                                    self.schema_out.relations))
        block_modules = {}
        if type(input_dim) == dict:
            self.input_dim = input_dim
        else:
            self.input_dim = {rel.id: input_dim for rel in self.schema.relations}
        if type(output_dim) == dict:
            self.output_dim = output_dim
        else:
            self.output_dim = {rel.id: output_dim for rel in self.schema_out.relations}
        for relation_i, relation_j in self.relation_pairs:
            block_module = SparseMatrixEquivariantLayerBlock(self.input_dim[relation_i.id],
                                                             self.output_dim[relation_j.id],
                                                              relation_i, relation_j)
            block_modules[str((relation_i.id, relation_j.id))] = block_module
        self.block_modules = nn.ModuleDict(block_modules)
        self.logger = logging.getLogger()
        #self.cache = {}            

    def forward(self, data, indices_identity=None, indices_transpose=None):
        data_out = SparseMatrixData(self.schema_out)
        for relation_i, relation_j in self.relation_pairs:
            #self.logger.warning("Relation: ({}, {})".format(relation_i.id, relation_j.id))
            X_in = data[relation_i.id]
            Y_in = data[relation_j.id]
            layer = self.block_modules[str((relation_i.id, relation_j.id))]
            indices_id = indices_identity[relation_i.id, relation_j.id]
            indices_trans = indices_transpose[relation_i.id, relation_j.id]
            Y_out = layer.forward(X_in, Y_in, indices_id, indices_trans)
            if relation_j.id not in data_out:
                data_out[relation_j.id] = Y_out
            else:
                data_out[relation_j.id] = data_out[relation_j.id] + Y_out
        return data_out


class SparseMatrixEntityPoolingLayer(SparseMatrixEquivariantLayer):
    def __init__(self, schema, input_dim=1, output_dim=1, entities=None):
        '''
        input_dim: either a rel_id: dimension dict, or an integer for all relations
        output_dim: either a rel_id: dimension dict, or an integer for all relations
        '''
        if entities == None:
            entities = schema.entities
        enc_relations = [Relation(i, [entity, entity], is_set=True)
                                for i, entity in enumerate(entities)]
        encodings_schema = DataSchema(entities, enc_relations)
        super().__init__(schema, input_dim, output_dim,
                         schema_out=encodings_schema)

    def forward(self, data, data_target=None):
        data_out = Data(self.schema_out)
        for relation_i, relation_j in self.relation_pairs:
            #self.logger.warning("Relation: ({}, {})".format(relation_i.id, relation_j.id))
            X_in = data[relation_i.id]
            Y_in = data_target[relation_j.id]
            layer = self.block_modules[str((relation_i.id, relation_j.id))]
            Y_out = layer.forward(X_in, Y_in, None, None)
            if relation_j.id not in data_out:
                data_out[relation_j.id] = Y_out
            else:
                data_out[relation_j.id] = data_out[relation_j.id] + Y_out
        return data_out
    