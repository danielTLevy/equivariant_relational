#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 20:18:51 2020

@author: Daniel
"""

class DataSchema():
    def __init__(self, entities, relations):
        self.entities = entities
        self.relations = relations
        
class Entity():
    def __init__(self, entity_number, n_instances):
        self.entity_number = entity_number
        self.n_instances = n_instances

    def __gt__(self, other):
        return self.entity_number > other.entity_number
    
    
class Relation():
    def __init__(self, entities):
        self.entities = entities