# -*- coding: utf-8 -*-
from collections import Counter
import torch
import numpy as np
import pdb

class SparseTensor:
    '''
    '''
    #TODO: want dense channel dimension, included as first dimension of values

    
    def __init__(self, indices, values, shape):
        assert len(shape) == indices.shape[0], "Number of dimensions in shape and indices do not match"
        assert indices.shape[1] == values.shape[0], "Number of nonzero elements in indices and values do not match"
        # array of n-dimensional indices
        self.indices = indices
        # Array of values
        self.values = values
        # Numpy array of dimension
        self.shape = shape

    @classmethod
    def from_tensor(cls, sparse_tensor):
        '''
        Initialize from pytorch's built-in sparse tensor
        '''
        indices = sparse_tensor.indices()
        values = sparse_tensor.values()
        shape = np.array(sparse_tensor.shape)
        return cls(indices, values, shape)

    def ndimension(self):
        return self.indices.shape[0]

    def nnz(self):
        return self.indices.shape[1]
    
    def to(self, *args, **kwargs):
        self.indices = self.indices.to(*args, **kwargs)
        self.values = self.values.to(*args, **kwargs)
        return self


    def add_sparse_tensors(self, other):
        assert self.shape == other.shape, "Mismatching shapes"
        combined_indices = torch.cat(self.indices, other.indices, dim=1)
        combined_values = torch.cat(self.values, other.values, dim=1)
        combined_tensor = torch.sparse_coo_tensor(combined_indices, combined_values,
                                size=tuple(self.shape)).coalesce()
        indices_out = combined_tensor.indices()
        values_out = combined_tensor.values()

        return SparseTensor(indices_out, values_out, self.shape)

    def add(self, other):
        if type(other) is SparseTensor:
            return self.add_sparse_tensors

    def __add__(self, other):
        return self.add(other)

    def __matmul__(self, other):
        values_out = self.values @ other
        return SparseTensor(self.indices, values_out, self.shape)

    def permute(self, permutation):
        shape_out = self.shape[permutation]
        indices_out = self.indices[permutation]    
        return SparseTensor(indices_out, self.values, shape_out)

    def permute_(self, permutation):
        '''
        In-place version of permute
        '''
        self.shape = self.shape[permutation]
        self.indices = self.indices[permutation]
        return self
    
    def transpose(self, dim1, dim2):
        indices_out = self.indices.clone()
        shape_out = self.shape.copy()

        tmp = indices_out[dim1].clone()
        tmp_dim = shape_out[dim1].clone()

        indices_out[dim1] = indices_out[dim2]
        shape_out[dim1] = shape_out[dim2]

        indices_out[dim2] = tmp
        shape_out[dim2] = tmp_dim
        return SparseTensor(indices_out, self.values, shape_out)

    def diagonal(self, offset=0, dim1=0, dim2=1):
        '''
        Returns a partial view of input with the its diagonal elements with
        respect to dim1 and dim2 appended as a dimension at the end of the shape.
        Requires dim1 < dim2
        
        offset is a useless parameter to match the method signature of a normal tensor
        '''
        assert dim1 < dim2, "Requires dim1 < dim2"
        assert self.shape[dim1] == self.shape[dim2], "dim1 and dim2 are not of equal length"

        # Get only values and indices where dim1 == dim2
        diag_idx = torch.where(self.indices[dim1] == self.indices[dim2])[0]
        diag_values = self.values[diag_idx]
        indices_out = torch.index_select(self.indices, 1, diag_idx)

        # Remove diagonal dimensions and append to end
        reshape_indices = torch.arange(self.ndimension() + 1)
        reshape_indices = reshape_indices[reshape_indices != dim1]
        reshape_indices = reshape_indices[reshape_indices != dim2]
        reshape_indices[-1] = dim1
        
        return SparseTensor(indices_out, diag_values, self.shape).permute_(reshape_indices)

    def pool(self, pooling_dims):
        remaining_dims = sorted(list(set(range(self.ndimension())) - set(pooling_dims)))
        pooled_shape = self.shape[remaining_dims]
        pooling_indices = self.indices[remaining_dims, :]
        pooled_tensor = torch.sparse_coo_tensor(pooling_indices, self.values,
                                                size=tuple(pooled_shape)).coalesce()
        pooled_indices = pooled_tensor.indices()
        pooled_values = pooled_tensor.values()

        return SparseTensor(pooled_indices, pooled_values, pooled_shape)


    def broadcast(self, indices_out_matching, indices_out_broadcast, new_dim_size):
        '''
        Assume self is already coalesced and that indices are sorted
        Add new dimension to end and expand by new_dim
        Use sparsity of intersection of indices and indices_out_matching, and 
        sparsity of indices_out_broadcast for the new dimension
        '''            
        def get_masks_of_intersection(array1, array2):
            # Return the mask of values of indices of array2 that intersect with array1
            # For example a1 = [0, 1, 2, 5], a2 = [1, 3, 2, 4], then intersection = [1, 2]
            # and array1_intersection_mask = [False, True, True, False]
            # and array2_intersection_mask = [True, False, True, False]
            n_in = array1.shape[1]
            combined = torch.cat((array1, array2), dim=1)
            intersection, intersection_idx, counts = combined.unique(return_counts=True, return_inverse=True, dim=1)
            intersection_mask = (counts > 1).T[intersection_idx].T
            array1_intersection_mask = intersection_mask[:n_in]
            array2_intersection_mask = intersection_mask[n_in:]
            return array1_intersection_mask, array2_intersection_mask
            
        # Get unique values of matching output indices, as well as a way to reverse this process
        indices_out_m_unique, indices_out_m_inverse_idx = indices_out_matching.unique(dim=1, return_inverse=True)
        
        # Get masks of which values to take from self.indices and from the unique matching output indices
        indices_intersection_mask, indices_out_m_unique_intersection_mask = get_masks_of_intersection(self.indices, indices_out_m_unique)

        # Get mask of which values to take from the matching output indices by undoing uniqueness operation
        indices_out_m_intersection_mask = indices_out_m_unique_intersection_mask[indices_out_m_inverse_idx]

        # Get matching output indices
        indices_out_m_intersection = indices_out_matching.T[indices_out_m_intersection_mask].T
        
        # Get corresponding indices of dimension to broadcast to
        indices_out_broadcast = indices_out_broadcast.T[indices_out_m_intersection_mask].T.unsqueeze(0)
        indices_out = torch.cat([indices_out_m_intersection, indices_out_broadcast])
        
        
        # Get values to take, repeating values to broadcast
        values_intersection = self.values[indices_intersection_mask]
        _, value_counts = torch.unique_consecutive(indices_out_m_intersection, dim=1, return_counts=True)
        values_out = torch.repeat_interleave(values_intersection, value_counts)
        
        # Get new shape:
        shape_out = np.concatenate((self.shape, [new_dim_size]))
        
        return SparseTensor(indices_out, values_out, shape_out)
    
    def diag_embed(self, indices_tgt):
        '''
        indices_tgt: (2, nnz) target indices of dimensions to embed diagonal onto
        Embed the source tensor onto two highest dimensions
        '''
        indices_out = self.indices[-1]
        torch.cat(indices)
        torch.cat(indices_out)
        #TODO: complete this
        return SparseTensor(indices_out, self.values, self.shape)

    def to_dense(self):
        out = torch.zeros(tuple(self.shape), dtype=self.values.dtype)
        return out.index_put_(tuple(self.indices), self.values)



if __name__ == '__main__':
    
            
    def test_broadcast():
        values = torch.tensor([1., 3., 5., 7., 11.])
        indices_in = torch.LongTensor([[0,0], [1,0], [1,2], [3, 0], [3, 3]]).T
        X = SparseTensor(indices_in, values, np.array([4, 4]))
        
        indices_out = torch.LongTensor([[0,1,2], [0,2,1], [0,3,2], [1, 1, 1], [1, 1, 2], [1, 2, 2], [1, 3, 0]]).T
        intersection = torch.LongTensor([[1, 2], [3, 0]]).T
        expected_out = torch.LongTensor([[1, 2, 0], [1, 2, 1], [3, 0, 1]]).T
        assert (X.broadcast(indices_out[1:], indices_out[0], 2).indices == expected_out).all()
    
    test_broadcast()
    
    def same_values(sparse_data, val_array):
        return Counter(sparse_data.values.numpy()) == Counter(tuple(val_array))


    '''
    1ooo
    2o3o
    oooo
    4oo5
    '''
    values1 = torch.tensor([1., 2., 3., 4., 5.])
    indices1 = torch.LongTensor([[0,0], [1,0], [1,2], [3, 0], [3, 3]]).T
    shape1 = np.array([4, 4])
    X = SparseTensor(indices1, values1, shape1)
    
    
    
    '''
    oooo    oooo
    oo1o    o45o
    o2oo    oo6o
    oo3o    7ooo
    '''
    indices2 = torch.LongTensor([[0,1,2], [0,2,1], [0,3,2], [1, 1, 1], [1, 1, 2], [1, 2, 2], [1, 3, 0]]).T
    values2 = torch.tensor([1., 2., 3., 4., 5., 6., 7.])
    shape2 = np.array([2, 4, 4])
    Y = SparseTensor(indices2, values2, shape2)

    Yp = Y.pool([1])
    '''
    o 2 4 o
    7 4 11 o
    '''
    assert same_values(Yp, [4, 2, 7, 4, 11])
    '''
    
    '''
    Yd = Y.diagonal(0, 1, 2)
    assert same_values(Yd, [4, 6])

    Xb = X.broadcast(Y.indices[1:], Y.indices[0], Y.shape[0])
    assert Xb.nnz() == 3
    assert same_values(Xb, [3, 3, 4])

