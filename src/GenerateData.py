# -*- coding: utf-8 -*-

# Schema taken from EERN paper

import numpy as np
import torch
from DataSchema import DataSchema, Entity, Relation

class SchoolGenerator():
    def __init__(self, n_student, n_course, n_professor):
        self.n_student = n_student
        self.n_course = n_course
        self.n_professor = n_professor

        ent_students = Entity(0, self.n_student)
        ent_courses = Entity(1, self.n_course)
        ent_professors = Entity(2, self.n_professor)
        entities = [ent_students, ent_courses, ent_professors]

        #TODO: Fix student self-relation to have two channels
        
        relations = []
        #Takes
        relations.append(Relation(0, [ent_students, ent_courses], 1))
        #Reference
        relations.append(Relation(1, [ent_students, ent_professors], 1))
        #Teaches
        relations.append(Relation(2, [ent_professors, ent_courses], 1))
        #Prereq
        relations.append(Relation(3, [ent_courses, ent_courses], 1))
        #Student
        relations.append(Relation(4, [ent_students], 1))
        #Course
        relations.append(Relation(5, [ent_courses], 1))
        #Professor
        relations.append(Relation(6, [ent_professors], 1))
        
        # pick n dimensions
        # Draw from n-dimensional normal dist to get encodings for each entity
        self.schema = DataSchema(entities, relations)
        self.embeddings = None

    def generate_embeddings(self, n_dim_ent=5, batch_size=1):
        np.random.seed(0)
        embed_students = np.random.normal(size=(batch_size, self.n_student, n_dim_ent))
        embed_courses = np.random.normal(size=(batch_size, self.n_course, n_dim_ent))
        embed_professors = np.random.normal(size=(batch_size, self.n_professor, n_dim_ent))
        self.embeddings = [embed_students, embed_courses, embed_professors]

    def generate_data(self, n_dim_ent=5, batch_size=1):
        if self.embeddings == None:
            self.generate_embeddings(n_dim_ent, batch_size)
        
        # TODO: make two-channeled
        def rel_student_fn(embedding):
            return 100*np.mean(np.abs(np.sin(embedding)), 1)
        
        def rel_courses_fn(embedding):
            return 100*np.round(np.sum(np.arctan(np.exp(embedding)), 1))
        
        def rel_professor_fn(embedding):
            return np.sum(np.sign(embedding), 1) + n_dim_ent
        
        def rel_takes_fn(embedding_student, embedding_course):
            return 100/(1 + np.exp(embedding_student @ embedding_course.T))
        
        def rel_ref_fn(embedding_student, embedding_professor):
            return np.sign(embedding_student @ embedding_professor.T)
        
        def rel_teaches_fn(embed_professor, embed_course):
            return 50 + 50*(np.sin(embed_professor) @ np.cos(embed_course).T)
        
        def rel_prereq_fn(embed_course1, embed_course2):
            return 50*np.pi*np.arctan(embed_course1 @ embed_course2.T)
        
        
        rel_fns = {
                0: rel_takes_fn, 1: rel_ref_fn, 2: rel_teaches_fn, 3: rel_prereq_fn,
                4: rel_student_fn, 5: rel_courses_fn, 6: rel_professor_fn}
    
        # TODO: change sparsity
        data = {}
        for relation in self.schema.relations:
            entities = relation.entities
            relation_data = torch.zeros(batch_size, 1, *relation.get_shape())
            for batch in range(batch_size):
                ent_embeddings  = [self.embeddings[ent.id][batch] for ent in entities]
                relation_data[batch] =  torch.tensor(rel_fns[relation.id](*ent_embeddings))
            data[relation.id] = relation_data
        return data
