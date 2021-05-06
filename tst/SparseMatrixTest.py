# -*- coding: utf-8 -*-


import unittest
import numpy as np
import torch
from src.SparseMatrix import SparseMatrix
import pdb

class TestSparseMatrix(unittest.TestCase):
    def setUp(self):
        '''
        1ooo
        2o3o
        oooo
        4oo5
        '''
        values1 = torch.arange(1, 6, dtype=torch.float32).view(5, 1)
        indices1 = torch.LongTensor([[0,0], [1,0], [1,2], [3, 0], [3, 3]]).T
        shape1 =(4, 4, 1)
        self.X = SparseMatrix(indices1, values1, shape1)        

        '''
        o1o2
        oo3o
        oo4o
        5ooo
        '''
        values2 = torch.arange(1, 6, dtype=torch.float32).view(5, 1)
        indices2 = torch.LongTensor([[0,1], [0,3], [1,2], [2, 2], [3, 0]]).T
        shape2 =(4, 4, 1)
        self.Y = SparseMatrix(indices2, values2, shape2) 
        
        self.pooled = torch.arange(1, 5, dtype=torch.float32).view(4, 1)
    

        # Two-channeled versions:
        '''
        1ooo 6ooo
        2o3o 7o8o
        oooo oooo
        4oo5 9ooX
        '''
        values1_2 = torch.arange(1, 11, dtype=torch.float32).view(2, 5).T
        indices1 = torch.LongTensor([[0,0], [1,0], [1,2], [3, 0], [3, 3]]).T
        shape1_2 =(4, 4, 2)
        self.X2 = SparseMatrix(indices1, values1_2, shape1_2)        

        '''
        o1o2 o6o7
        oo3o oo8o
        oo4o oo9o
        5ooo Xooo
        '''
        values2_2 = torch.arange(1, 11, dtype=torch.float32).view(2, 5).T
        indices2 = torch.LongTensor([[0,1], [0,3], [1,2], [2, 2], [3, 0]]).T
        shape2_2 =(4, 4, 2)
        self.Y2 = SparseMatrix(indices2, values2_2, shape2_2) 
        
        self.pooled2 = torch.arange(1, 9, dtype=torch.float32).view(2, 4).T
    
    def assertSameValues(self, values, expected_values):
        expected_tensor = torch.Tensor(expected_values)
        self.assertTrue(values.allclose(expected_tensor))

    def assertSameTensor(self, sparse_matrix, val_array):
        # Assert values are the same, regardless of order
        unique_vals, val_counts = np.unique(sparse_matrix.values.numpy(), return_counts=True)
        unique_vals_exp, val_counts_exp = np.unique(val_array, return_counts=True)
        
        self.assertTrue((unique_vals == unique_vals_exp).all() and (val_counts == val_counts_exp).all())

class TestSparseTensorPool(TestSparseMatrix):
    def test_pool_row(self):
        Xp = self.X.pool('row')
        self.assertSameValues(Xp, np.array([[7, 0, 3, 5]]).T)
        
        Xp2 = self.X2.pool('row')
        self.assertSameValues(Xp2, np.array([[7, 0, 3, 5],[22, 0, 8, 10]]).T)

    def test_pool_col(self):
        Xp = self.X.pool('col')
        self.assertSameValues(Xp, np.array([[1, 5, 0, 9]]).T)
        
        Xp2 = self.X2.pool('col')
        self.assertSameValues(Xp2, np.array([[1, 5, 0, 9], [6, 15, 0, 19]]).T)
    
    def test_pool_diag(self):
        Xp = self.X.pool('diag')
        self.assertSameValues(Xp, np.array([6.]))
        Xp2 = self.X2.pool('diag')
        self.assertSameValues(Xp2, np.array([6.,16]))

    def test_pool_all(self):
        Xp = self.X.pool('all')
        self.assertSameValues(Xp, np.array([15.]))
        Xp2 = self.X2.pool('all')
        self.assertSameValues(Xp2, np.array([15.,40.]))

class TestSparseMatrixDiagonal(TestSparseMatrix):
    def test_diagonal(self):
        Xd = self.X.gather_diag()
        self.assertSameValues(Xd, np.array([[1, 0, 0, 5]]).T)
        
    def test_diagonal_multichannel(self):
        Xd = self.X2.gather_diag()
        self.assertSameValues(Xd, np.array([[1, 0, 0, 5],[6,0,0,10]]).T)

class TestSparseMatrixBroadcast(TestSparseMatrix):
    def test_broadcast_col(self):
        zero_matrix = SparseMatrix.from_other_sparse_matrix(self.X, n_channels=1)
        out = zero_matrix.broadcast(self.pooled, "col")
        '''
        1ooo
        2o2o
        oooo
        4oo4
        '''
        self.assertSameValues(out.values, np.array([[1,2,2,4,4]]).T)

    def test_broadcast_row(self):
        zero_matrix = SparseMatrix.from_other_sparse_matrix(self.X, n_channels=1)
        out = zero_matrix.broadcast(self.pooled, "row")
        '''
        1ooo
        1o3o
        oooo
        1oo4
        '''
        self.assertSameValues(out.values, np.array([[1,1,3,1,4]]).T)

    def test_broadcast_all(self):
        zero_matrix = SparseMatrix.from_other_sparse_matrix(self.X, n_channels=1)
        out = zero_matrix.broadcast(torch.Tensor([5.]), "all")
        '''
        5ooo
        5o5o
        oooo
        5oo5
        '''
        self.assertSameValues(out.values, np.array([[5,5,5,5,5]]).T)
        
    def test_broadcast_diag(self):
        zero_matrix = SparseMatrix.from_other_sparse_matrix(self.X, n_channels=1)
        out = zero_matrix.broadcast(torch.Tensor([5., 2]), "diag")
        '''
        5ooo
        0o0o
        oooo
        0oo5
        '''
        self.assertSameValues(out.values, np.array([[5,0,0,0,5],[2,0,0,0,2]]).T)

class TestSparseMatrixEmbedDiag(TestSparseMatrix):
    def test_embed_diag(self):
        indices_diag = torch.LongTensor([0, 4])
        self.assertTrue(torch.equal(self.X.indices_diag, indices_diag))
        
        data = torch.Tensor(np.array([[1, 3, 5, 6]]).T)
        expected_values = np.array([[1, 0, 0 ,0, 6]]).T
        embed_diag = self.X.embed_diag(data)
        self.assertSameValues(embed_diag.values, expected_values)

    def test_embed_diag_multidim(self):        
        data = torch.Tensor(np.array([[1, 3, 5, 7], [2, 4, 6, 8], [11, 12, 13, 14]]).T)
        expected_values = np.array([[1, 0, 0 ,0, 7],[2,0,0,0,8], [11, 0, 0, 0, 14]]).T
        embed_diag = self.X.embed_diag(data)
        self.assertSameValues(embed_diag.values, expected_values)

class TestSparseMatrixIdentity(TestSparseMatrix):
    def test_calc_intersection_mask(self):
        mask_X_expected = torch.BoolTensor([False, False, True, True, False])
        mask_Y_expected = torch.BoolTensor([False, False, True, False, True])
        mask_X, mask_Y  = self.X.calc_intersection_mask(self.Y)

        self.assertTrue(torch.equal(mask_X, mask_X_expected))
        self.assertTrue(torch.equal(mask_Y, mask_Y_expected))
        
        mask_Y_2, mask_X_2  = self.Y.calc_intersection_mask(self.X)
        self.assertTrue(torch.equal(mask_X_2, mask_X_expected))
        self.assertTrue(torch.equal(mask_Y_2, mask_Y_expected))


    def test_identity(self):
        mask_X, mask_Y  = self.X.calc_intersection_mask(self.Y)
        Y_identity = self.Y.identity()
        self.assertTrue(Y_identity.equal(self.Y))
        
        # Y's values + idx, with values where X overlaps with Y
        Y_identity_X = self.Y.identity(mask_X)
        self.assertSameValues(Y_identity_X.values, np.array([[0, 0, 3, 4, 0]]).T)
        
        # X's values + idx, with values where X overlaps with Y
        X_identity_Y = self.X.identity(mask_Y)
        self.assertSameValues(X_identity_Y.values, np.array([[0, 0, 3, 0, 5]]).T)
        
        '''
        #TODO: fill in values
        # Y's values, X's idx, with values where X overlaps with Y
        Y_identity_X_X = self.Y.identity(mask_Y, mask_X, self.X.indices)
        self.assertSameValues(Y_identity_X_X.values, np.array([[0, 0, 3, 4, 0]]).T)

        # X's values, Y's idx, with values where X overlaps with Y
        X_identity_Y_Y = self.Y.identity(mask_Y, mask_X, self.X.indices)
        self.assertSameValues(X_identity_Y_Y.values, np.array([[0, 0, 3, 4, 0]]).T)
        '''
    
    def test_gather_identity(self):
        mask_X, mask_Y  = self.X.calc_intersection_mask(self.Y)
        '''
        X    Y     X.id(Y) Y.id(X)
        1ooo o1o2  o0o0    0ooo
        2o3o oo3o  oo3o    0o3o
        oooo oo4o  oo0o    oooo
        4oo5 5ooo  4ooo    5oo0
        '''
        data_X = self.X.gather_mask(mask_X)
        Y_out = self.Y.broadcast_from_mask(data_X, mask_Y)
        self.assertSameValues(Y_out.values, np.array([[0,0,3,0,4]]).T)
        
        data_Y = self.Y.gather_mask(mask_Y)
        X_out = self.X.broadcast_from_mask(data_Y, mask_X)
        self.assertSameValues(X_out.values, np.array([[0,0,3,5,0]]).T)
        

    



class TestSparseMatrixTranspose(TestSparseMatrix):
    def test_transpose_self(self):
        Y_trans = self.Y.transpose()
        self.assertSameValues(Y_trans.values, np.array([[5, 1, 3, 4, 2]]).T)
        '''
        Y=   Y.T=  Y.T(X)=
        o1o2 ooo5 0ooo 
        oo3o 1ooo 1o0o
        oo4o o34o oooo
        5ooo 2ooo 2oo0
        '''

        X_trans = self.X.transpose()
        self.assertSameValues(X_trans.values, np.array([[1, 2, 4, 3, 5]]).T)
        '''
        X=   X.T  X.T(Y)=
        1ooo 12o4 o2o4
        2o3o oooo oo0o
        oooo o3oo oo0o
        4oo5 ooo5 0ooo
        '''

    def test_transpose_intersection_mask(self):
        
        intersection_mask = self.Y.calc_transpose_intersection_overlap(self.X)
        mask_Y_expected = torch.BoolTensor([False, True, False, False, True])
        mask_X_expected = torch.BoolTensor([False, True, False, True, False])
        self.assertTrue(torch.equal(intersection_mask[0], mask_Y_expected))
        self.assertTrue(torch.equal(intersection_mask[1], mask_X_expected))
        
    def test_transpose_other(self):
        intersection_mask = self.Y.calc_transpose_intersection_overlap(self.X)
        Y_trans_X = self.Y.transpose(intersection_mask[0])
        self.assertSameValues(Y_trans_X.values, np.array([[0, 1, 0, 0, 2]]).T)

    def test_gather_transpose(self):
        Y_T_mask, X_mask = self.Y.calc_transpose_intersection_overlap(self.X)
        # Values of Y that overlap with X
        Y_transpose_data = self.Y.gather_transpose(Y_T_mask)
        # Values of Y.T with Indices of X 
        Y_trans_X = self.X.broadcast_from_mask(Y_transpose_data, X_mask)
        self.assertSameValues(Y_trans_X.values, np.array([[0, 1, 0, 2, 0]]).T)
        
        # Values of X that overlap with Y
        X_T_mask, Y_mask = self.X.calc_transpose_intersection_overlap(self.Y)

        X_transpose_data = self.X.gather_transpose(X_T_mask)
        # Values of X.T with indices of Y 
        X_trans_Y = self.Y.broadcast_from_mask(X_transpose_data, Y_mask)
        self.assertSameValues(X_trans_Y.values, np.array([[2, 4, 0, 0, 0]]).T)

if __name__ == '__main__':
    unittest.main()    

