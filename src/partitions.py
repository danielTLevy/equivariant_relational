#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 02:24:51 2020

@author: Daniel
"""
#%%
import numpy as np
import torch 
import itertools
from DataSchema import DataSchema, Entity, Relation

#%%
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
        entity_indices = 1 + np.where(combined_entities == entity)[0]
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
                    if entity_index < 1 + relation_in_length:
                        inputs.append(entity_index)
                    else:
                        outputs.append(entity_index - relation_in_length)
                mapping.append((set(inputs), set(outputs)))
        output.append(mapping)
    return output


if __name__ == '__main__': 
    #Example 
    ##Ri = {n1, n2, n3}
    #Rj = {m1, m2}
    X = torch.tensor(np.arange(12)).view(2,2,3)
    
    
    # Entity index : number instances mapping
    entities = [Entity(0, 5), Entity(1, 2), Entity(2, 3)]
    relation_i = Relation(0, [entities[1], entities[0], entities[2], entities[0]])
    relation_j = Relation(1, [entities[0], entities[1]])
    relations = [relation_i, relation_j]
    data_schema = DataSchema(entities, relations)
    
    input_output_partitions = get_all_input_output_partitions(data_schema, relation_i, relation_j)
    assert(len(input_output_partitions) == 1*2*5)