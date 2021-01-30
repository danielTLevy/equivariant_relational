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
    def __init__(self, entity_id, n_instances):
        self.entity_id = entity_id
        self.n_instances = n_instances

    def __gt__(self, other):
        return self.entity_id > other.entity_id
    
    def __repr__(self):
        return "Entity(entity_id={}, n_instances={})".format(self.entity_id, self.n_instances)
    
class Relation():
    def __init__(self, relation_id, entities, n_channels=1):
        self.relation_id = relation_id
        self.entities = entities
        self.n_channels = n_channels

    def get_shape(self):
        return [entity.n_instances for entity in self.entities]

    def __repr__(self):
        return "Relation(relation_id={}, entities={})".format(self.relation_id, self.entities)