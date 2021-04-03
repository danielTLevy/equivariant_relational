# -*- coding: utf-8 -*-


import unittest
import numpy as np
import torch
from src.SparseTensor import SparseTensor
from collections import Counter
import pdb

class TestSparseTensor(unittest.TestCase):
    def setUp(self):
        '''
        1ooo
        2o3o
        oooo
        4oo5
        '''
        values1 = torch.arange(1, 6, dtype=torch.float32).view(1, 5)
        indices1 = torch.LongTensor([[0,0], [1,0], [1,2], [3, 0], [3, 3]]).T
        shape1 = np.array([4, 4])
        self.X = SparseTensor(indices1, values1, shape1)

        '''
        oooo    oooo
        oo1o    o45o
        o2oo    oo6o
        oo3o    7ooo
        '''
        indices2 = torch.LongTensor([[0,1,2], [0,2,1], [0,3,2], [1, 1, 1], [1, 1, 2], [1, 2, 2], [1, 3, 0]]).T
        values2 = torch.arange(1, 8, dtype=torch.float32).view(1,7)
        shape2 = np.array([2, 4, 4])
        self.Y = SparseTensor(indices2, values2, shape2)
        
        '''
        1 2 0 3
        '''
        indices3 = torch.LongTensor([[0, 1, 3]])
        values3 = torch.arange(1, 4, dtype=torch.float32).view(1,3)
        shape3 = np.array([4])
        self.Z = SparseTensor(indices3, values3, shape3)

        '''
        1 0 
        '''
        self.W = SparseTensor(torch.LongTensor([[0]]), torch.Tensor([[1.]]), np.array([2]))
        
        
    def assertSame(self, sparse_tensor, val_array):
        # Assert values are the same, regardless of order
        unique_vals, val_counts = np.unique(sparse_tensor.values.numpy(), return_counts=True)
        unique_vals_exp, val_counts_exp = np.unique(val_array, return_counts=True)
        
        self.assertTrue((unique_vals == unique_vals_exp).all() and (val_counts == val_counts_exp).all())

class TestSparseTensorPool(TestSparseTensor):
    def test_pool(self):
        Yp = self.Y.pool([1])
        '''
        o 2 4 o
        7 4 11 o
        '''
        self.assertSame(Yp, [[4, 2, 7, 4, 11]])

class TestSparseTensorDiagonal(TestSparseTensor):
    def test_diagonal(self):
        Yd = self.Y.diagonal(0, 1, 2)
        self.assertSame(Yd, [[4, 6]])

class TestSparseTensorPermute(TestSparseTensor):
    pass

class TestSparseTensorBroadcast(TestSparseTensor):
    def test_broadcast_one_dim(self):
        Xb = self.X.broadcast([self.Y.shape[0]], self.Y.indices[0:1], self.Y.indices[1:])
        self.assertEquals(Xb.nnz(), 3)
        self.assertSame(Xb, [[3, 3, 4]])
    
    def test_broadcast_two_dim(self):
        Wb = self.W.broadcast([self.Y.shape[1]]*2, self.Y.indices[1:], self.Y.indices[0:1],)
        self.assertSame(Wb, [[1, 1, 1]])

    def test_broadcast_zero_dim(self):
        Xb = self.X.broadcast([], self.X.indices[0:0], self.X.indices)
        self.assertSame(Xb, self.X.values.numpy())

    def test_broadcast_no_matching_dim(self):
        Xb = self.X.broadcast([self.Y.shape[0]],  self.Y.indices[0:1], self.Y.indices[1:1])
        pdb.set_trace()
        
        self.assertSame(self.same_values(Xb, [1,2]))

if __name__ == '__main__':
    unittest.main()    

