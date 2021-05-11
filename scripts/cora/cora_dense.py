# -*- coding: utf-8 -*-

'''
Entities:
    paper: 2708: Each has unique int. Nonconsecutive, so must translate to idx
    word: 1433. Each is like "word125" so just need to convert to 125 to get idx
    class: 7. Must translate text to idx 
Relationships:
    cites: 5429, paper to paper
    content: 49216, paper to word
    paper: 2708, paper to class
    Alternatively, could make class a channel attribute for a paper?
    This would involve getting the code working for sets too probably
Goal is to predict class - use 10-fold cross validation.
Use 8/10 for train, 1/10 for val, 1/10 for test – train on val before testing on test 
'''

from src.DataSchema import DataSchema, Entity, Relation, Data
from src.EquivariantNetwork import EquivariantNetwork
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import csv
import pdb

csv_file_str = '../data/cora/{}.csv'

def load_data():
    paper_names = []
    classes = []
    word_names = ['word'+str(i+1) for i in range(1433)]

    with open(csv_file_str.format('paper')) as paperfile:
        reader = csv.reader(paperfile)
        for paper_name, class_name in reader:
            paper_names.append(paper_name)
            classes.append(class_name)

    class_names = list(np.unique(classes))
    class_name_to_idx = {class_name : i for i, class_name in enumerate(class_names)}
    paper_name_to_idx = {paper_name: i for i, paper_name in enumerate(paper_names)}
    paper = np.array([[paper_name_to_idx[paper_name] for paper_name in paper_names],
                      [class_name_to_idx[class_name] for class_name in classes]])

    cites = []
    with open(csv_file_str.format('cites')) as citesfile:
        reader = csv.reader(citesfile)
        for citer, citee in reader:
            cites.append([paper_name_to_idx[citer], paper_name_to_idx[citee]])
    cites = np.array(cites).T

    content = []
    def word_to_idx(word):
        '''
        words all formatted like: "word1328"
        '''
        return int(word[4:]) - 1

    with open(csv_file_str.format('content')) as contentfile:
        reader = csv.reader(contentfile)
        for paper_name, word_name in reader:
            content.append([paper_name_to_idx[paper_name],
                            word_to_idx(word_name)])
    content = np.array(content).T

    n_papers = len(paper_names)
    n_classes = len(class_names)
    n_words = len(word_names)
    ent_papers = Entity(0, n_papers)
    ent_classes = Entity(1, n_classes)
    ent_words = Entity(2, n_words)
    entities = [ent_papers, ent_classes, ent_words]
    rel_paper = Relation(0, [ent_papers, ent_classes])
    rel_cites = Relation(1, [ent_papers, ent_papers])
    rel_content = Relation(2, [ent_papers, ent_words])
    relations = [rel_paper, rel_cites, rel_content]
    schema = DataSchema(entities, relations)

    class_targets = torch.LongTensor(paper[1])

    paper_matrix = torch.zeros(n_papers, n_classes)
    paper_matrix[paper] = 1
    
    cites_matrix = torch.zeros(n_papers, n_papers)
    cites_matrix[cites] = 1
    
    content_matrix = torch.zeros(n_papers, n_words)
    content_matrix[content] = 1
    
    

    data = Data(schema)
    data[0] = paper_matrix.unsqueeze(0).unsqueeze(0)
    data[1] = cites_matrix.unsqueeze(0).unsqueeze(0)
    data[2] = content_matrix.unsqueeze(0).unsqueeze(0)
    return data, schema, class_targets


#%%
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, schema, targets = load_data()
    data = data.to(device)   
    targets = targets.to(device)
    relations = schema.relations

    #%%

    # Loss functions:        
    def sparse_loss_fcn(loss, data_pred, data_true):
        '''
        Loss over all relations
        '''
        loss.zero_()
        
        for relation in schema.relations:
            loss += F.mse_loss(data_pred[relation.id], data_true[relation.id])
        loss = loss / len(relations)
        return loss

    def classification_loss(data_pred, data_true):
        return F.cross_entropy(data_pred, data_true)


    net = EquivariantNetwork(schema, 1).to(device)

    learning_rate = 1e-5
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.0, 0.999))

    sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                 factor=0.5,
                                                 patience=10,
                                                 verbose=True)
    #%%
    epochs= 50
    progress = tqdm(range(epochs), desc="Loss: ", position=0, leave=True)
    loss = torch.zeros(1).to(device)
    for i in progress:
        optimizer.zero_grad()
        data_out = net.forward(data)
        train_loss = classification_loss(loss, data_out, targets)
        train_loss.backward()
        optimizer.step()
        progress.set_description("Train: {:.4f}".format(train_loss.item()))
