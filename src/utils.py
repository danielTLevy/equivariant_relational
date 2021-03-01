#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 22:31:50 2021

@author: Daniel
"""

import numpy as np
import itertools

# Non-relation dimensions (batch and channel)
PREFIX_DIMS = 2

# https://stackoverflow.com/questions/19368375/set-partitions-in-python/30134039
def get_partitions(collection):
    if len(collection) == 1:
        yield [ collection ]
        return
    first = collection[0]
    for smaller in get_partitions(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset 
        yield [ [ first ] ] + smaller
    

def get_all_entity_partitions(data_schema, combined_entities):
    '''
    
    Returns: List of all mappings between entities and partitioning of their 
    indicies in the concatenated input and output
    '''
    all_entities = np.unique(combined_entities)

    # Map of entity: index
    partitions = {}
    for entity in all_entities:
        entity_indices = PREFIX_DIMS + np.where(combined_entities == entity)[0]
        partitions[entity] = list(get_partitions(list(entity_indices)))
    
    partition_product = itertools.product(*partitions.values())
    entity_partitions = []
    for combination in partition_product:
        entity_partition_map = {}
        for i, entity in enumerate(all_entities):
            entity_partition_map[entity] = combination[i]
        entity_partitions.append(entity_partition_map)
    return entity_partitions

def get_all_input_output_partitions(data_schema, relation_in, relation_out):
    '''
    Returns: a list of all "input output partitions", which are tuple pairs
    of the set of indices in the input and the set of indices in the output
    that are equal to each other.
    '''
    entities_in = relation_in.entities
    entities_out = relation_out.entities
    combined_entities = np.array(entities_in + entities_out)
    entity_partitions = get_all_entity_partitions(data_schema, combined_entities)
    relation_in_length = len(relation_in.entities)

    output = []
    for entity_partition in entity_partitions:
        mapping = []  
        for partitions in entity_partition.values():
            for partition in partitions:
                # get indices with respect to input and output instead of
                # with respect to their concatenation
                inputs = []
                outputs = []
                for entity_index in partition:
                    if entity_index < PREFIX_DIMS + relation_in_length:
                        inputs.append(entity_index)
                    else:
                        outputs.append(entity_index - relation_in_length)
                mapping.append((set(inputs), set(outputs)))
        output.append(mapping)
    return output


def update_observed(observed_old, p_keep, min_observed):
    '''
    Updates observed entries
    Source: https://github.com/drgrhm/exch_relational/blob/master/util.py#L186
    Parameters:
        observed_old: old binary array of observed entries
        p_keep: proportion of observed entries to keep
        min_observed: minimum number of entries per row and column
    Returns:
        observed_new: updated array of observed entries
    '''

    inds_sc = np.array(np.nonzero(observed_old)).T

    n_keep = int(p_keep * inds_sc.shape[0])
    n_drop = inds_sc.shape[0] - n_keep

    inds_sc_keep = np.concatenate( (np.ones(n_keep), np.zeros(n_drop)) )
    np.random.shuffle(inds_sc_keep)
    inds_sc = inds_sc[inds_sc_keep == 1, :]

    observed_new = np.zeros_like(observed_old)
    observed_new[inds_sc[:,0], inds_sc[:,1]] = 1

    shape = observed_new.shape
    rows = np.sum(observed_new, axis=1)
    for i in np.array(range(shape[0]))[rows < min_observed]:
        diff = observed_old[i, :] - observed_new[i, :]
        resample_inds = np.array(range(shape[1]))[diff == 1]
        jj = np.random.choice(resample_inds, int(min_observed - rows[i]), replace=False)
        observed_new[i, jj] = 1

    cols = np.sum(observed_new, axis=0)
    for j in np.array(range(shape[1]))[cols < min_observed]:
        diff = observed_old[:, j] - observed_new[:, j]
        resample_inds = np.array(range(shape[0]))[diff == 1]
        ii = np.random.choice(resample_inds, int(min_observed - cols[j]), replace=False)
        observed_new[ii, j] = 1

    return observed_new
