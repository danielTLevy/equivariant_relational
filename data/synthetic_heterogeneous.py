# -*- coding: utf-8 -*-
from src.DataSchema import DataSchema, Entity, Relation, SparseMatrixData
from src.SparseMatrix import SparseMatrix
import numpy as np
import torch

class SyntheticHG:
    def __init__(self, n_ents=1, n_rels=1, embed_dim=10,
                 n_instances=1000, sparsity=0.01, p_het=0,
                 gen_links='uniform'):
        self.embed_dim = embed_dim
        self.n_instances = n_instances
        self.sparsity = sparsity
        self.p_het = p_het
        if gen_links == 'proportional':
            self.generate_links = self.generate_links_proportional
        else:
            self.generate_links = self.generate_links_uniform
        # Create schema
        entities = []
        for ent_i in range(n_ents):
            entities.append(Entity(ent_i, n_instances))
        relations = {}
        for rel_id in range(n_rels):
            ent_i = np.random.randint(0, n_ents)
            ent_j = np.random.randint(0, n_ents)
            relations[rel_id] = Relation(rel_id, [entities[ent_i], entities[ent_j]], 1, is_set=False)
        self.schema = DataSchema(entities, relations)
        self.data = SparseMatrixData(self.schema)
        # Dict tracking whether each relation is homophilic or heterophilic
        self.rel_het = {}
        # Create entity embeddings
        self.ent_embed = {}
        for ent_i in range(n_ents):
            self.ent_embed[ent_i] = np.random.normal(0., 1., (n_instances, embed_dim))

        # Use entity embeddings to create data matrix
        for rel_id in range(n_rels):
            embed_i = self.ent_embed[relations[rel_id].entities[0].id]
            embed_j = self.ent_embed[relations[rel_id].entities[1].id]
            # Choose whether to make relation homophilic or heterophilic
            het = np.random.rand(1)[0] < p_het
            self.rel_het[rel_id] = het
            sample_indices = self.generate_links(embed_i, embed_j, sparsity, het)
            sample_indices_tensor = torch.LongTensor(sample_indices)
            sample_values_tensor = torch.ones((sample_indices.shape[1], 1))
            data_matrix = SparseMatrix(sample_indices_tensor, sample_values_tensor, \
                                       shape=(n_instances, n_instances, 1)).coalesce()
            self.data[rel_id] = data_matrix

    def generate_links_uniform(self, embed_i, embed_j, sparsity, het=True):
        '''
        Generate links with uniform probabilities, but only where the
        dot product between entities is positive (or negative, for heterophilic)
        '''
        embed_i_norm = np.linalg.norm(embed_i, 2, 1)
        embed_j_norm = np.linalg.norm(embed_j, 2, 1)
        costh = np.dot(embed_i, embed_j.T)/np.outer(embed_i_norm, embed_j_norm)
        all_links_mask = costh > 0
        if het:
            all_links_mask = ~all_links_mask

        # Uniformly randomly sparsify data matrices
        all_links_indices = np.array(all_links_mask.nonzero())
        n_links = all_links_indices.shape[1]

        #p_links = all_links_indices.shape[0] / all_links_mask.size
        n_sample = int(sparsity*all_links_mask.size)
        if n_sample < n_links:
            sample_indices_idx = np.random.choice(range(n_links), n_sample)
            sample_indices = all_links_indices[:, sample_indices_idx]
        return sample_indices

    def generate_links_proportional(self, embed_i, embed_j, sparsity, het=True):
        '''
        Generate links with probabilites proportional to logit of dot
        product bewteen entity embeddings.
        '''
        dot = np.dot(embed_i, embed_j.T)
        if het:
            dot = -dot
        dot_prob_unnorm = 1/(1 + np.exp(-dot))
        # No self-links
        np.fill_diagonal(dot_prob_unnorm, 0)
        n_sample = int(sparsity*dot.size)
        # Generate link with probability proportional to logitss
        dot_prob = n_sample*dot_prob_unnorm/dot_prob_unnorm.sum()
        links_mask = dot_prob > np.random.random_sample(dot_prob.shape)
        links_indices = np.array(links_mask.nonzero())
        return links_indices

    def label_random_weight(self, embeds, n_classes, embed_dim):
        class_weights = np.random.rand(n_classes, embed_dim)
        labels = np.argmax(class_weights @ embeds.T, axis=0)
        return labels

    def label_nearest_neighbour(self, embeds, n_classes, embed_dim):
        # Generate cores for each class
        random_cores = np.random.normal(0., 1., (n_classes, embed_dim))
        embed_repeat = np.repeat(embeds[:, :, np.newaxis], n_classes, axis=2)
        # Calculate L2-distance to each core
        embed_diff = embed_repeat - random_cores.T
        embed_dist = np.linalg.norm(embed_diff, axis=1)
        # Get closest core
        labels = np.argmax(embed_dist, 1)
        return labels


    def make_node_classification_task(self, n_classes=3, p_test=0.2,
                                      p_val=0.2, labelling='weight'):
        self.n_classes = n_classes
        target_node_type = 0
        embeds = self.ent_embed[target_node_type]
        if labelling == 'weight':
            labelling_fn = self.label_random_weight
        elif labelling == 'neighbour':
            labelling_fn = self.label_nearest_neighbour
        labels = labelling_fn(embeds, n_classes, self.embed_dim)

        np.random.seed(0)
        shuffled_indices = np.arange(self.n_instances)
        np.random.shuffle(shuffled_indices)

        n_test = int(p_test*self.n_instances)
        n_val = int(p_val*self.n_instances)
        self.test_idx = shuffled_indices[:n_test]
        self.val_idx = shuffled_indices[n_test:n_test+n_val]
        self.train_idx = shuffled_indices[n_test+n_val:]
        target_ent = self.schema.entities[target_node_type]
        self.schema_out = DataSchema([target_ent],
                                {0: Relation(0, [target_ent, target_ent],is_set=True)})
        self.labels = labels
        self.data_target = SparseMatrixData(self.schema_out)
        self.data_target[0] = SparseMatrix(
            indices = torch.arange(self.n_instances, dtype=torch.int64).repeat(2,1),
            values=torch.zeros([self.n_instances, n_classes]),
            shape=(self.n_instances, self.n_instances, n_classes),
            is_set=True)

