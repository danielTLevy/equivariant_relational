#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import torch
import torch.nn as nn
import itertools
import logging
from src.utils import get_all_ops, PREFIX_DIMS
from src.DataSchema import Data
from src.SparseMatrix import SparseMatrix
import pdb

LOG_LEVEL = logging.ERROR

class SparseMatrixEquivariantLayerBlock(nn.Module):
    # Layer mapping between two relations
    def __init__(self, input_dim, output_dim, relation_in, relation_out):
        super(SparseMatrixEquivariantLayerBlock, self).__init__()
        assert len(relation_in.entities) == 2, "Relation must be second order"
        assert len(relation_out.entities) == 2, "Relation must be second order"
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.all_ops = get_all_ops(relation_in, relation_out)
        self.n_params = len(self.all_ops)
        stdv = 0.1 / math.sqrt(self.in_dim)
        self.weights = nn.Parameter(torch.Tensor(self.n_params, self.in_dim, self.out_dim).uniform_(-stdv, stdv))
        self.bias = nn.Parameter(torch.ones(self.n_params))
        self.output_shape = [0, self.out_dim] + [entity.n_instances for entity in relation_out.entities]
        self.logger = logging.getLogger()
        self.logger.setLevel(LOG_LEVEL)

    def output_op(self, op_str, X_out, data):
        op, index_str = op_str.split('_')
        if op == 'b':
            return X_out.broadcast(data, index_str)
        elif op == 'e':
            return X_out.embed_diag(data)

    def input_op(self, op_str, X_in):        
        op, index_str = op_str.split('_')
        if op == 'g':
            return X_in.gather_diag()
        elif op == 'p':
            return X_in.pool(index_str)

    def forward(self, X_in, X_out, indices_identity, indices_trans):
        '''
        X_in: Source sparse tensor
        X_out: Correpsonding sparse tensor for target relation
        '''
        self.logger.info("n_params: {}".format(self.n_params))
        Y = SparseMatrix.from_other_sparse_matrix(X_out, self.out_dim)
        #TODO: can add a cache for input operations here
        for i in range(self.n_params):
            op_inp, op_out = self.all_ops[i]
            if op_out[0] == "i":
                # Identity
                X_intersection_vals = X_in.gather_mask(indices_identity[0])
                X_mul = X_intersection_vals @ self.weights[i]
                X_op_out = X_out.broadcast_from_mask(X_mul, indices_identity[1])
            elif op_out[0] == "t":
                # Transpose
                X_T_intersection_vals = X_in.gather_transpose(indices_trans[0])
                X_mul = X_T_intersection_vals @ self.weights[i]
                X_op_out = X_out.broadcast_from_mask(X_mul, indices_trans[1])
            else:
                # Pool or Gather or Do Nothing
                X_op_inp = self.input_op(op_inp, X_in)
                # Multiply values by weight
                X_mul = torch.matmul(X_op_inp, self.weights[i])
                # Broadcast or Embed Diag or Transpose
                X_op_out = self.output_op(op_out, X_out, X_mul)
            assert X_op_out.nnz() == X_out.nnz()
            assert Y.nnz() == X_out.nnz(), "Y: {}, X_out: {}".format(Y.nnz(), X_out.nnz())
            assert Y.nnz() == X_op_out.nnz(), "Y: {}, X_op_out: {}".format(Y.nnz(), X_op_out.nnz())
            Y = Y + X_op_out + self.bias[i]
        return Y


class SparseMatrixEquivariantLayer(nn.Module):
    def __init__(self, data_schema, input_dim=1, output_dim=1):
        super(SparseMatrixEquivariantLayer, self).__init__()
        self.data_schema = data_schema
        self.relation_pairs = list(itertools.product(self.data_schema.relations,
                                                self.data_schema.relations))
        block_modules = []
        self.input_dim = input_dim
        self.output_dim = output_dim
        for relation_i, relation_j in self.relation_pairs:
            block_module = SparseMatrixEquivariantLayerBlock(self.input_dim, self.output_dim,
                                                              relation_i, relation_j)
            block_modules.append(block_module)
        self.block_modules = nn.ModuleList(block_modules)
        self.logger = logging.getLogger()
        #self.cache = {}            

    def forward(self, data, indices_identity=None, indices_transpose=None):
        data_out = Data(self.data_schema)
        for i, (relation_i, relation_j) in enumerate(self.relation_pairs):
            self.logger.info("Relation: ({}, {})".format(relation_i.id, relation_j.id))
            X_in = data[relation_i.id]
            Y_in = data[relation_j.id]
            layer = self.block_modules[i]
            indices_id = indices_identity[relation_i.id, relation_j.id]
            indices_trans = indices_transpose[relation_i.id, relation_j.id]
            Y_out = layer.forward(X_in, Y_in, indices_id, indices_trans)
            if relation_j.id not in data_out:
                data_out[relation_j.id] = Y_out
            else:
                data_out[relation_j.id] = data_out[relation_j.id] + Y_out
        return data_out