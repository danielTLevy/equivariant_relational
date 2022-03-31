# -*- coding: utf-8 -*-
from src.DataSchema import DataSchema, Entity, Relation, SparseMatrixData
from src.SparseMatrix import SparseMatrix
import numpy as np
from collections import defaultdict

import scipy.sparse as sp
from scipy.stats import ortho_group, special_ortho_group
import random
import torch

class SyntheticHG:
    def __init__(self, n_ents=1, n_rels=1, embed_dim=10,
                 n_instances=1000, sparsity=0.01, p_het=0,
                 gen_links='uniform', schema_str=''):
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
        # Override schema with manually entered schema if provided
        if schema_str != '':
            relations = {}
            schema_tuples = eval(schema_str)
            for rel_id, (ent_i, ent_j) in enumerate(schema_tuples):
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

        # Keep copy of this data
        self.full_data = self.data
        self.full_schema = self.schema
        self.flat = False

    def to_flat(self):
        tot_instances = sum([ent.n_instances for ent in self.schema.entities])
        tot_entity = Entity(0, tot_instances)
        tot_rel = Relation(0, [tot_entity, tot_entity])

        data_shifted = {}
        for rel_id, rel in self.schema.relations.items():
            ent_i, ent_j = rel.entities[0], rel.entities[1]
            data_clone = self.data[rel_id].clone()
            data_clone.n = tot_instances
            data_clone.m = tot_instances
            data_clone.indices[0] += self.shift(ent_i)
            data_clone.indices[1] += self.shift(ent_j)
            data_shifted[rel_id] = data_clone

        # Sparse Matrix containing all data
        data_diag = SparseMatrix(indices=torch.arange(tot_instances).expand(2, tot_instances),
                                 values=torch.ones((tot_instances, 1)),
                                 shape=(tot_instances, tot_instances, 1))
        data_full = data_diag.clone()
        for rel_id, data_matrix in data_shifted.items():
            data_full = data_full + data_matrix

        data_out = SparseMatrix.from_other_sparse_matrix(data_full, 0)
        # Load up all edge data
        for rel_id, data_rel in data_shifted.items():
            data_rel_full = SparseMatrix.from_other_sparse_matrix(data_full, 1) + data_rel
            data_out.values = torch.cat([data_out.values, data_rel_full.values], 1)
            data_out.n_channels += 1

        # Update with new schema
        self.schema = DataSchema([tot_entity], {0: tot_rel})
        self.n_instances = tot_instances
        self.data = SparseMatrixData(self.schema)
        self.data[0] = data_out
        self.flat = True
        return self.data

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
            sample_indices_idx = np.random.choice(range(n_links), n_sample, replace=False)
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
        if self.flat:
            raise NotImplementedError()
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

    def make_link_prediction_task(self, p_test=0.2, p_val=0.2,
                                  val_neg_type='random', tail_weighted=False):
        self.target_rel_id = 0
        if tail_weighted:
            self.tail_prob = self.make_tail_prob()
        if self.flat:
            target_data = self.full_data[self.target_rel_id]
        else:
            target_data = self.data[self.target_rel_id]
        self.target_indices = target_data.indices.cpu().numpy()
        n_idx = self.target_indices.shape[1]
        n_test = int(p_test*n_idx)
        n_val = int(p_val*n_idx)
        all_pos_idx = np.arange(n_idx)
        np.random.shuffle(all_pos_idx)
        train_pos_idx = all_pos_idx[n_test+n_val:]
        self.train_pos = self.target_indices[:, train_pos_idx]

        val_pos_idx = all_pos_idx[n_test:n_test+n_val]
        self.valid_pos = self.target_indices[:, val_pos_idx]

        test_pos_idx = all_pos_idx[:n_test]
        self.test_pos =  self.target_indices[:, test_pos_idx]

    def get_train_valid_pos(self):
        return self.train_pos, self.valid_pos

    def get_train_neg(self, tail_weighted=False):
        '''
        Training negative samples have the same head nodes as the positive
        samples, but with random tail nodes. If tail_weighted, then tail nodes
        are weighted by their frequency in the positive samples
        '''
        if not self.flat:
            start_t = 0
        else:
            ent = self.full_schema.relations[self.target_rel_id].entities[1]
            start_t = self.shift(ent)
        t_arange = np.arange(start_t + self.n_instances)
        neg_h =  self.train_pos[0]
        n_pos = len(neg_h)
        if tail_weighted:
            neg_t = list(np.random.choice(t_arange, size=n_pos, p=self.tail_prob))
        else:
            neg_t = list(np.random.choice(t_arange, size=n_pos))
        return np.array([neg_h, neg_t])

    def get_valid_neg(self, val_neg='random'):
        if val_neg == '2hop':
            # NOTE: doesn't work
            self.neg_neigh = self.make_2hop()
            return self.get_valid_neg_2hop()
        elif val_neg == 'randomtw':
            return self.get_valid_neg_uniform(tail_weighted=True)
        else:
            return self.get_valid_neg_uniform(tail_weighted=False)

    def get_valid_neg_uniform(self, tail_weighted=False):
        if not self.flat:
            start_t = 0
        else:
            ent = self.full_schema.relations[self.target_rel_id].entities[0]
            start_t = self.shift(ent)
        t_arange = np.arange(start_t + self.n_instances)
        '''get neg_neigh'''
        neg_h = self.valid_pos[0]
        n_pos = len(neg_h)
        if tail_weighted:
            neg_t = list(np.random.choice(t_arange, size=n_pos, p=self.tail_prob))
        else:
            neg_t = list(np.random.choice(t_arange, size=n_pos))
        return np.array([neg_h, neg_t])


    def make_2hop(self):
        '''
        Make neg_neigh, a dict of head:tail dicts for each relation, giving
        all non-positive 2 hop neighbours for each node
        NOTE: this currently doesn't work.'
        '''
        # Get full adjacency matrix
        pos_links = 0
        # Add training links
        for rel_i, rel in self.schema.relations.items():
            offset_i = self.shift(rel.entities[0])
            offset_j = self.shift(rel.entities[1])
            data_matrix = self.data[rel_i].clone()
            data_matrix.indices[0, :] += offset_i
            data_matrix.indices[1, :] += offset_j
            # Doesn't work with offset:
            sparse_matrix = data_matrix.to_scipy_sparse()
            pos_links += sparse_matrix + sparse_matrix.T

        # Add in validation linksd
        rel = self.schema.relations[self.target_rel_id]
        offset_i = self.shift(rel.entities[0])
        offset_j = self.shift(rel.entities[1])
        values = [1] * self.valid_pos.shape[1]
        valid_pos_offset = self.valid_pos.copy()
        valid_pos_offset[0,:] += offset_i
        valid_pos_offset[1,:] += offset_j
        valid_of_rel = sp.coo_matrix(values, valid_pos_offset, shape=pos_links.shape)
        pos_links += valid_of_rel
        # Square of adjacency matrix
        r_double_neighs = np.dot(pos_links, pos_links)
        data = r_double_neighs.data
        data[:] = 1
        # 2-hop-neighs = (A^2 - A - I)
        r_double_neighs = \
            sp.coo_matrix((data, r_double_neighs.nonzero()), shape=np.shape(pos_links), dtype=int) \
            - sp.coo_matrix(pos_links, dtype=int) \
            - sp.lil_matrix(np.eye(np.shape(pos_links)[0], dtype=int))
        data = r_double_neighs.data
        pos_count_index = np.where(data > 0)
        row, col = r_double_neighs.nonzero()
        r_double_neighs = sp.coo_matrix((data[pos_count_index], (row[pos_count_index], col[pos_count_index])),
                                        shape=np.shape(pos_links))

        row, col = r_double_neighs.nonzero()
        data = r_double_neighs.data
        sec_index = np.where(data > 0)
        row, col = row[sec_index], col[sec_index]

        total_nodes = self.n_instances*self.len(self.schema.entities)
        relation_range = [self.shift(ent) for ent in self.schema.entities] + [total_nodes]

        h_type, t_type = rel.entities[0].id, rel.entities[1].id
        # Get all examples of this relation by checking row and col ranges
        r_id_index = np.where((row >= relation_range[h_type]) & (row < relation_range[h_type + 1])
                              & (col >= relation_range[t_type]) & (col < relation_range[t_type + 1]))[0]

        r_row, r_col = row[r_id_index], col[r_id_index]
        neg_neigh = defaultdict(list)
        for h_id, t_id in zip(r_row, r_col):
            neg_neigh[h_id].append(t_id)
        neg_neigh[h_id].append(t_id)
        return neg_neigh

    def get_valid_neg_2hop(self):
        '''get pos_neigh'''
        pos_neigh = defaultdict(list)
        row, col = self.valid_pos
        for h_id, t_id in zip(row, col):
            pos_neigh[h_id].append(t_id)

        '''sample neg as same number as pos for each head node'''
        valid_neigh = [[], []]
        for h_id in sorted(list(pos_neigh.keys())):
            n_pos = len(pos_neigh[h_id])

            neg_list = random.choices(self.neg_neigh[h_id], k=n_pos) if len(
                self.neg_neigh[h_id]) != 0 else []
            valid_neigh[0].extend([h_id] * len(neg_list))
            valid_neigh[1].extend(neg_list)
        return valid_neigh

    def get_test_neigh_2hop(self):
        return self.get_test_neigh()

    def get_test_neigh(self, test_rel_id):
        random.seed(1)
        neg_neigh, pos_neigh, test_neigh, test_label = dict(), dict(), dict(), dict()
        edge_types = self.test_types
        '''get sec_neigh'''
        # Get full adjacency matrix
        pos_links = 0
        for r_id in self.schema.relations:
            pos_links += self.links['data'][r_id] + self.links['data'][r_id].T
        for r_id in self.links_test['data'].keys():
            pos_links += self.links_test['data'][r_id] + self.links_test['data'][r_id].T
        for r_id in self.valid_pos.keys():
            values = [1] * len(self.valid_pos[r_id][0])
            valid_of_rel = sp.coo_matrix((values, self.valid_pos[r_id]), shape=pos_links.shape)
            pos_links += valid_of_rel
        # Square of adjacency matrix
        r_double_neighs = np.dot(pos_links, pos_links)
        data = r_double_neighs.data
        data[:] = 1
        # 2-hop-nieghs = (A^2 - A - I)
        r_double_neighs = \
            sp.coo_matrix((data, r_double_neighs.nonzero()), shape=np.shape(pos_links), dtype=int) \
            - sp.coo_matrix(pos_links, dtype=int) \
            - sp.lil_matrix(np.eye(np.shape(pos_links)[0], dtype=int))
        data = r_double_neighs.data
        pos_count_index = np.where(data > 0)
        row, col = r_double_neighs.nonzero()
        r_double_neighs = sp.coo_matrix((data[pos_count_index], (row[pos_count_index], col[pos_count_index])),
                                        shape=np.shape(pos_links))

        row, col = r_double_neighs.nonzero()
        data = r_double_neighs.data
        sec_index = np.where(data > 0)
        row, col = row[sec_index], col[sec_index]

        relation_range = [self.nodes['shift'][k] for k in range(len(self.nodes['shift']))] + [self.nodes['total']]
        for r_id in self.links_test['data'].keys():
            neg_neigh[r_id] = defaultdict(list)
            h_type, t_type = self.links_test['meta'][r_id]
            # Get all examples of this relation by checking row and col ranges
            r_id_index = np.where((row >= relation_range[h_type]) & (row < relation_range[h_type + 1])
                                  & (col >= relation_range[t_type]) & (col < relation_range[t_type + 1]))[0]
            # r_num = np.zeros((3, 3))
            # for h_id, t_id in zip(row, col):
            #     r_num[self.get_node_type(h_id)][self.get_node_type(t_id)] += 1
            r_row, r_col = row[r_id_index], col[r_id_index]
            for h_id, t_id in zip(r_row, r_col):
                neg_neigh[r_id][h_id].append(t_id)

        for r_id in edge_types:
            '''get pos_neigh'''
            pos_neigh[r_id] = defaultdict(list)
            (row, col), data = self.links_test['data'][r_id].nonzero(), self.links_test['data'][r_id].data
            for h_id, t_id in zip(row, col):
                pos_neigh[r_id][h_id].append(t_id)

            '''sample neg as same number as pos for each head node'''
            test_neigh[r_id] = [[], []]
            pos_list = [[], []]
            test_label[r_id] = []
            for h_id in sorted(list(pos_neigh[r_id].keys())):
                pos_list[0] = [h_id] * len(pos_neigh[r_id][h_id])
                pos_list[1] = pos_neigh[r_id][h_id]
                test_neigh[r_id][0].extend(pos_list[0])
                test_neigh[r_id][1].extend(pos_list[1])
                test_label[r_id].extend([1] * len(pos_list[0]))

                neg_list = random.choices(neg_neigh[r_id][h_id], k=len(pos_list[0])) if len(
                    neg_neigh[r_id][h_id]) != 0 else []
                test_neigh[r_id][0].extend([h_id] * len(neg_list))
                test_neigh[r_id][1].extend(neg_list)
                test_label[r_id].extend([0] * len(neg_list))
        return test_neigh, test_label

    def get_test_neigh_w_random(self):
        random.seed(1)
        all_had_neigh = defaultdict(list)
        neg_neigh, pos_neigh, test_neigh, test_label = dict(), dict(), dict(), dict()
        edge_types = self.test_types
        '''get pos_links of train and test data'''
        pos_links = 0
        for r_id in self.links['data'].keys():
            pos_links += self.links['data'][r_id] + self.links['data'][r_id].T
        for r_id in self.links_test['data'].keys():
            pos_links += self.links_test['data'][r_id] + self.links_test['data'][r_id].T
        for r_id in self.valid_pos.keys():
            values = [1] * len(self.valid_pos[r_id][0])
            valid_of_rel = sp.coo_matrix((values, self.valid_pos[r_id]), shape=pos_links.shape)
            pos_links += valid_of_rel

        row, col = pos_links.nonzero()
        for h_id, t_id in zip(row, col):
            all_had_neigh[h_id].append(t_id)
        for h_id in all_had_neigh.keys():
            all_had_neigh[h_id] = set(all_had_neigh[h_id])
        for r_id in edge_types:
            h_type, t_type = self.links_test['meta'][r_id]
            t_range = (self.nodes['shift'][t_type], self.nodes['shift'][t_type] + self.nodes['count'][t_type])
            '''get pos_neigh and neg_neigh'''
            pos_neigh[r_id], neg_neigh[r_id] = defaultdict(list), defaultdict(list)
            (row, col), data = self.links_test['data'][r_id].nonzero(), self.links_test['data'][r_id].data
            for h_id, t_id in zip(row, col):
                pos_neigh[r_id][h_id].append(t_id)
                neg_t = random.randrange(t_range[0], t_range[1])
                while neg_t in all_had_neigh[h_id]:
                    neg_t = random.randrange(t_range[0], t_range[1])
                neg_neigh[r_id][h_id].append(neg_t)
            '''get the test_neigh'''
            test_neigh[r_id] = [[], []]
            pos_list = [[], []]
            neg_list = [[], []]
            test_label[r_id] = []
            for h_id in sorted(list(pos_neigh[r_id].keys())):
                pos_list[0] = [h_id] * len(pos_neigh[r_id][h_id])
                pos_list[1] = pos_neigh[r_id][h_id]
                test_neigh[r_id][0].extend(pos_list[0])
                test_neigh[r_id][1].extend(pos_list[1])
                test_label[r_id].extend([1] * len(pos_neigh[r_id][h_id]))
                neg_list[0] = [h_id] * len(neg_neigh[r_id][h_id])
                neg_list[1] = neg_neigh[r_id][h_id]
                test_neigh[r_id][0].extend(neg_list[0])
                test_neigh[r_id][1].extend(neg_list[1])
                test_label[r_id].extend([0] * len(neg_neigh[r_id][h_id]))
        return test_neigh, test_label

    def get_test_neigh_full_random(self):
        edge_types = self.test_types
        random.seed(1)
        '''get pos_links of train and test data'''
        all_had_neigh = defaultdict(list)
        pos_links = 0
        for r_id in self.links['data'].keys():
            pos_links += self.links['data'][r_id] + self.links['data'][r_id].T
        for r_id in self.links_test['data'].keys():
            pos_links += self.links_test['data'][r_id] + self.links_test['data'][r_id].T
        for r_id in self.valid_pos.keys():
            values = [1] * len(self.valid_pos[r_id][0])
            valid_of_rel = sp.coo_matrix((values, self.valid_pos[r_id]), shape=pos_links.shape)
            pos_links += valid_of_rel

        row, col = pos_links.nonzero()
        for h_id, t_id in zip(row, col):
            all_had_neigh[h_id].append(t_id)
        for h_id in all_had_neigh.keys():
            all_had_neigh[h_id] = set(all_had_neigh[h_id])
        test_neigh, test_label = dict(), dict()
        for r_id in edge_types:
            test_neigh[r_id] = [[], []]
            test_label[r_id] = []
            h_type, t_type = self.links_test['meta'][r_id]
            h_range = (self.nodes['shift'][h_type], self.nodes['shift'][h_type] + self.nodes['count'][h_type])
            t_range = (self.nodes['shift'][t_type], self.nodes['shift'][t_type] + self.nodes['count'][t_type])
            (row, col), data = self.links_test['data'][r_id].nonzero(), self.links_test['data'][r_id].data
            for h_id, t_id in zip(row, col):
                test_neigh[r_id][0].append(h_id)
                test_neigh[r_id][1].append(t_id)
                test_label[r_id].append(1)
                neg_h = random.randrange(h_range[0], h_range[1])
                neg_t = random.randrange(t_range[0], t_range[1])
                while neg_t in all_had_neigh[neg_h]:
                    neg_h = random.randrange(h_range[0], h_range[1])
                    neg_t = random.randrange(t_range[0], t_range[1])
                test_neigh[r_id][0].append(neg_h)
                test_neigh[r_id][1].append(neg_t)
                test_label[r_id].append(0)

        return test_neigh, test_label

    def make_tail_prob(self):
        r_id = self.target_rel_id
        entities = self.schema.relations[r_id].entities
        t_range = (0, entities[1].n_instances)
        node_to_count = {}
        tails, counts = np.unique( self.target_indices[1], return_counts=True)
        for node_i, count in zip(tails, counts):
            node_to_count[node_i] = count
        all_counts = []
        for tail_i in range(t_range[0], t_range[1]):
            all_counts.append(node_to_count.get(tail_i, 1))
        tail_probs = np.array(all_counts) / sum(all_counts)
        return tail_probs

    def shift(self, entity):
        return entity.n_instances*entity.id


