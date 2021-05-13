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
#%%
from src.DataSchema import DataSchema, Entity, Relation, SparseTensorData
from src.SparseTensor import SparseTensor
from src.EquivariantNetwork import SparseEquivariantNetwork
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import csv
import pdb
#%%
csv_file_str = './data/cora/{}.csv'

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

    # For each paper, get a random negative sample
    random_class_offset = np.random.randint(1, n_classes, (n_papers,))
    paper_neg = np.stack((paper[0], (paper[1] + random_class_offset) % n_classes))
    paper_matrix = SparseTensor(
            indices = torch.LongTensor(np.concatenate((paper, paper_neg),axis=1)),
            values = torch.cat((torch.ones(1, paper.shape[1]), torch.zeros(1, paper_neg.shape[1])), 1),
            shape = np.array([n_papers, n_classes])
            ).coalesce()

    class_targets = torch.LongTensor(paper[1])


    # Randomly fill in values and coalesce to remove duplicates
    cites_neg = np.random.randint(0, n_papers, cites.shape)
    cites_matrix = SparseTensor(
            indices = torch.LongTensor(np.concatenate((cites, cites_neg),axis=1)),
            values = torch.cat((torch.ones(1, cites.shape[1]), torch.zeros(1, cites_neg.shape[1])), 1),
            shape = np.array([n_papers, n_papers])
            ).coalesce()

    # For each paper, randomly fill in values and coalesce to remove duplicates
    content_neg = np.stack((np.random.randint(0, n_papers, (content.shape[1],)),
                            np.random.randint(0, n_words, (content.shape[1],))))
    content_matrix = SparseTensor(
            indices = torch.LongTensor(np.concatenate((content, content_neg),axis=1)),
            values = torch.cat((torch.ones(1, content.shape[1]), torch.zeros(1, content_neg.shape[1])), 1),
            shape = np.array([n_papers, n_words])
            ).coalesce()


    paper_dense_indices = np.array([np.tile(range(n_papers), n_classes),
                                np.repeat(range(n_classes), n_papers)])
    paper_dense_values = torch.zeros(paper_dense_indices.shape[1])
    for paper_i, class_name in enumerate(classes):
        class_i = class_name_to_idx[class_name]
        paper_dense_values[paper_i*n_classes + class_i] = 1
    paper_dense_matrix = SparseTensor(
            indices = torch.LongTensor(paper_dense_indices),
            values = torch.Tensor(paper_dense_values).unsqueeze(0),
            shape = np.array([n_papers, n_classes])
            )    

    data = SparseTensorData(schema)
    data[0] = paper_dense_matrix
    data[1] = cites_matrix
    data[2] = content_matrix
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

    def binary_loss(data_pred, data_true):
        '''
        Loss over all relations
        '''
        loss = torch.zeros(1).to(device)
        for relation in schema.relations:
            rel_id = relation.id
            loss += F.binary_cross_entropy(data_pred[rel_id].values, data_true[rel_id].values)
        loss = loss / len(relations)
        return loss

    def classification_loss(data_pred, data_true):
        return F.cross_entropy(data_pred, data_true)

    def binary_cross_entropy(data_pred, data_true):
        return F.binary_cross_entropy(data_pred, data_true)

    net = SparseEquivariantNetwork(schema, 1,  target_rel=0,
                                   final_activation=nn.Softmax(dim=1)).to(device)

    learning_rate = 1e-4
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.0, 0.999))

    sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                 factor=0.5,
                                                 patience=20,
                                                 verbose=True)
    #%%
    epochs= 10
    save_every = 10
    progress = tqdm(range(epochs), desc="Loss: ", position=0, leave=True)
    PATH = "../cora_tensor_model.pt"
    for i in progress:
        optimizer.zero_grad()
        data_out = net(data)
        train_loss = classification_loss(data_out, targets)
        train_loss.backward()
        optimizer.step()
        progress.set_description("Train: {:.4f}".format(train_loss.item()))
        sched.step(train_loss.item())
        if i % save_every == 0:
            torch.save({
                'epoch': i,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss.item()
                }, PATH)

    #%%
    '''
    checkpoint = torch.load("../cora_tensor_model.pt", map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['model_state_dict'])
    loss = checkpoint['loss']
    
    net.eval()
    data_out = net(data, indices_identity, indices_transpose)
    '''
