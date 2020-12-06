#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 02:24:51 2020

@author: Daniel
"""
#%%
import numpy as np
import torch 
import torch.nn as nn
import itertools

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
    

def get_all_entity_partitions(entities, n_instances, entities_in, entities_out):
    combined_entities = np.array(entities_in + entities_out)
    # Map of entity: index
    partitions = {}
    for entity in entities:
        entity_indices = np.where(combined_entities == entity)[0]
        partitions[entity] = list(get_partitions(list(entity_indices)))
    
    partition_product = itertools.product(*partitions.values())
    entity_partitions = [{entity: combination[i] for i, entity in enumerate(entities)} for combination in partition_product]
    return entity_partitions

def get_all_input_output_partitions(entity_partitions):
    output = []
    for entity_partition in entity_partitions:
        mapping = []  
        for partitions in entity_partition.values():
            for partition in partitions:
                inputs = []
                outputs = []
                for entity_index in partition:
                    if entity_index < len(entities_in):
                        inputs.append(entity_index)
                    else:
                        outputs.append(entity_index - len(entities_in))
                mapping.append((set(inputs), set(outputs)))
        output.append(mapping)
    return output

#%%
#Example 
##Ri = {n1, n2, n3}
#Rj = {m1, m2}
X = torch.tensor(np.arange(12)).view(2,2,3)


# Entity index : number instances mapping
entities = [0, 1, 2]
n_instances = {0:5, 1:2, 2:3}
# Input tensor entities
entities_in = [1, 0, 2, 0]
# Output tensor entities
entities_out = [0, 1]

entity_partitions = get_all_entity_partitions(entities, n_instances, entities_in, entities_out)
input_output_partitions = get_all_input_output_partitions(entity_partitions)
assert(len(input_output_partitions) == 10)