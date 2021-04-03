#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import numpy as np
import torch
from src.GenerateData import RandomSparseData
from src.SparseTensor import SparseTensor
from src.SparseEquivariantLayer import SparseEquivariantLayerBlock, SparseEquivariantLayer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%%
N_STUDENT = 50
N_COURSE = 30
N_PROFESSOR = 25
EMBEDDING_DIMS = 2
BATCH_SIZE = 1
SPARSITY = 0.01
CHANNELS_IN = 2
CHANNELS_OUT = 3

data_generator = RandomSparseData((N_STUDENT, N_COURSE, N_PROFESSOR),
                               sparsity=SPARSITY, n_channels=CHANNELS_IN)
sparse_data = data_generator.observed
schema = sparse_data.schema
relations = schema.relations


# The relations:
#Relation(0, [ent_students, ent_courses], 1)
#Relation(1, [ent_students, ent_professors], 1)
#Relation(2, [ent_professors, ent_courses], 1)
#Relation(3, [ent_students, ent_professors, ent_courses], 1)
#Relation(4, [ent_courses, ent_courses], 1)
#Relation(5, [ent_students], 1)
#Relation(6, [ent_students, ent_students, ent_students, ent_courses], 1)

#%% Test individual relation
'''
relation_in_idx = 6
relation_out_idx = 6

sparse_in = sparse_data[relation_in_idx]
sparse_out = sparse_data[relation_out_idx]
layer =  SparseEquivariantLayerBlock(CHANNELS_IN, CHANNELS_OUT, schema,
                                    relations[relation_in_idx], relations[relation_out_idx])
layer(sparse_in, sparse_out)
'''

#%% Test whole layer
layer = SparseEquivariantLayer(schema, input_dim=CHANNELS_IN, output_dim=CHANNELS_OUT)
layer(sparse_data)

