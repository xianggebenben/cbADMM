from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Reddit2, Flickr, PPI
from torch_geometric.data import DataLoader


import torch
from torch_sparse import SparseTensor

from common import gcn_norm
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T
import os
from configparser import ConfigParser
config = ConfigParser()
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    config.read(os.path.join(BASE_DIR, 'config.ini'), encoding='utf-8')
except:
    config.read('config.ini', encoding='utf-8')

device = torch.device(config['common']['device'])

# =================== onehot label ===================
def onehot(label, num_classes=None):
    """
    return the onehot label for mse loss
    """
    if num_classes == None:
        classes = set(np.unique(label.detach().cpu().numpy()))
    else:
        classes = set(np.linspace(0, num_classes-1, num_classes))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    x = list(map(classes_dict.get, label.detach().cpu().numpy()))
    label_onehot = np.array(x)
    label_onehot = torch.tensor(label_onehot, dtype=torch.float)
    return label_onehot.to(device)


class cora():
    def __init__(self):
        self.data = Planetoid(root='/tmp/cora', name='cora')[0]
        self.processed_dir = Planetoid(root='/tmp/cora', name='cora').processed_dir
        self.x =self.data.x.to('cpu')
        self.label = self.data.y
        self.num_classes = max(self.label) + 1
        self.num_features = self.data.x.size()[1]
        self.label_train, self.label_test = self.data.y[self.data.train_mask].to('cpu'), self.data.y[self.data.test_mask].to('cpu')
        self.label_train_onehot = onehot(self.label_train, num_classes=self.num_classes).to('cpu')
        self.edge_index = self.data.edge_index
        self.adj = gcn_norm(self.data.edge_index).to('cpu')
        self.train_mask, self.test_mask = self.data.train_mask, self.data.test_mask


class pubmed():
    def __init__(self):
        self.data = Planetoid(root='/tmp/PubMed', name='PubMed')[0]
        self.processed_dir = Planetoid(root='/tmp/PubMed', name='PubMed').processed_dir
        self.x =self.data.x.to('cpu')
        self.label = self.data.y
        self.num_classes = max(self.label) + 1
        self.num_features = self.data.x.size()[1]
        self.label_train, self.label_test = self.data.y[self.data.train_mask].to('cpu'), self.data.y[self.data.test_mask].to('cpu')
        self.label_train_onehot = onehot(self.label_train, num_classes=self.num_classes).to('cpu')
        self.adj = gcn_norm(self.data.edge_index).to('cpu')
        self.edge_index = self.data.edge_index
        self.train_mask, self.test_mask = self.data.train_mask, self.data.test_mask


class citeseer():
    def __init__(self):
        self.data = Planetoid(root='/tmp/citeseer', name='citeseer')[0]
        self.processed_dir = Planetoid(root='/tmp/citeseer', name='citeseer').processed_dir
        self.x =self.data.x.to('cpu')
        self.label = self.data.y
        self.num_classes = max(self.label) + 1
        self.num_features = self.data.x.size()[1]
        self.label_train, self.label_test = self.data.y[self.data.train_mask].to('cpu'), self.data.y[self.data.test_mask].to('cpu')
        self.label_train_onehot = onehot(self.label_train, num_classes=self.num_classes).to('cpu')
        self.adj = gcn_norm(self.data.edge_index).to('cpu')
        self.edge_index = self.data.edge_index
        self.train_mask, self.test_mask = self.data.train_mask, self.data.test_mask



class amazon_computers():
    def __init__(self):
        self.data = Amazon(root='/tmp/computers', name='computers')[0]
        self.processed_dir = Amazon(root='/tmp/computers', name='computers').processed_dir
        self.x =self.data.x.to('cpu')
        self.label = self.data.y
        self.num_classes = max(self.label) + 1
        self.num_features = self.data.x.size()[1]

        # split the dataset
        split_dataset(self)
        self.label_train, self.label_test = self.data.y[self.data.train_mask].to('cpu'), self.data.y[self.data.test_mask].to('cpu')
        self.label_train_onehot = onehot(self.label_train, num_classes=self.num_classes).to('cpu')
        self.adj = gcn_norm(self.data.edge_index).to('cpu')
        self.edge_index = self.data.edge_index


class amazon_photo():
    def __init__(self):
        self.data = Amazon(root='/tmp/photo', name='photo')[0]
        self.processed_dir = Amazon(root='/tmp/photo', name='photo').processed_dir
        self.x = self.data.x.to('cpu')
        self.label = self.data.y
        self.num_classes = max(self.label) + 1
        self.num_features = self.data.x.size()[1]
        # split the dataset
        split_dataset(self)
        self.label_train, self.label_test = self.data.y[self.data.train_mask].to('cpu'), self.data.y[self.data.test_mask].to('cpu')
        self.label_train_onehot = onehot(self.label_train, num_classes=self.num_classes).to('cpu')
        self.adj = gcn_norm(self.data.edge_index).to('cpu')
        self.edge_index = self.data.edge_index

class coauthor_cs():
    def __init__(self):
        self.data = Coauthor(root='/tmp/cs', name='cs')[0]
        self.processed_dir = Coauthor(root='/tmp/cs', name='cs').processed_dir
        self.x = self.data.x.to('cpu')
        self.label = self.data.y
        self.num_classes = max(self.label) + 1
        self.num_features = self.data.x.size()[1]
        # split the dataset
        split_dataset(self)
        self.label_train, self.label_test = self.data.y[self.data.train_mask].to('cpu'), self.data.y[
            self.data.test_mask].to('cpu')
        self.label_train_onehot = onehot(self.label_train, num_classes=self.num_classes).to('cpu')
        self.adj = gcn_norm(self.data.edge_index).to('cpu')
        self.edge_index = self.data.edge_index

class coauthor_physics():
    def __init__(self):
        self.data = Coauthor(root='/tmp/physics', name='physics')[0]
        self.processed_dir = Coauthor(root='/tmp/physics', name='physics').processed_dir
        self.x =self.data.x.to('cpu')
        self.label = self.data.y
        self.num_classes = max(self.label) + 1
        self.num_features = self.data.x.size()[1]
        # split the dataset
        split_dataset(self)
        self.label_train, self.label_test = self.data.y[self.data.train_mask].to('cpu'), self.data.y[self.data.test_mask].to('cpu')
        self.label_train_onehot = onehot(self.label_train, num_classes=self.num_classes).to('cpu')
        self.adj = gcn_norm(self.data.edge_index).to('cpu')
        self.edge_index = self.data.edge_index
        self.num_edges = self.data.num_edges
class flickr():
    def __init__(self):
        self.data = Flickr(root='/tmp/Flickr')[0]
        self.processed_dir = Flickr(root='/tmp/Flickr').processed_dir
        self.x =self.data.x.to('cpu')
        self.label = self.data.y
        self.num_classes = max(self.label) + 1
        self.num_features = self.data.x.size()[1]

        self.train_mask = self.data.train_mask
        self.test_mask = self.data.test_mask
        self.label_train, self.label_test = self.data.y[self.data.train_mask].to('cpu'), self.data.y[self.data.test_mask].to('cpu')
        self.label_train_onehot = onehot(self.label_train, num_classes=self.num_classes).to('cpu')
        self.adj = gcn_norm(self.data.edge_index)
        self.edge_index = self.data.edge_index
        self.num_edges = self.data.num_edges

class ppi():
    def __init__(self, class_list=None):
        root = './data/PPI'
        train_dataset = PPI(root, split='train')
        test_dataset = PPI(root, split='test')
        train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        for data in train_loader:
            self.data_train = data
        for data in test_loader:
            self.data_test = data
        self.processed_dir = PPI(root='./data/PPI').processed_dir
        self.adj_train = gcn_norm(self.data_train.edge_index, num_nodes=self.data_train.x.size()[0]).to(device)
        self.adj_test = gcn_norm(self.data_test.edge_index, num_nodes=self.data_test.x.size()[0]).to(device)
        self.num_classes = self.data_train.y.size()[1]
        self.data_train.x = self.data_train.x.to(device)
        self.data_test.x = self.data_test.x.to(device)
        self.data_train.y = self.data_train.y.to(device)
        self.data_test.y = self.data_test.y.to(device)

class reddit2():
    def __init__(self, class_list=None):
        self.data = Reddit2(root='tmp/Reddit2')[0]
        self.processed_dir = Reddit2(root='tmp/Reddit2').processed_dir
        self.x =self.data.x.to('cpu')
        self.label = self.data.y.to('cpu')
        if class_list is None:
            self.num_classes = max(self.data.y)+1
            self.num_features = self.data.x.size()[1]

            self.train_mask = self.data.train_mask
            self.test_mask = self.data.test_mask
            self.label_train, self.label_test = self.data.y[self.data.train_mask].to('cpu'), self.data.y[self.data.test_mask].to('cpu')
            self.adj = gcn_norm(self.data.edge_index).to('cpu')
            self.edge_index = self.data.edge_index.to('cpu')
            self.num_edges = self.data.num_edges
            self.label_train_onehot = onehot(self.label_train).to('cpu')
            self.data = None
        else:
            class_mask = torch.zeros(size=(1, self.label.size()[0])).squeeze(dim=0).bool().fill_(False)
            for class_name in class_list:
                class_idx = (self.label == class_name).nonzero().view(-1)
                class_mask[class_idx] = True
            self.label = self.label[class_mask]
            self.x = self.data.x[class_mask].to('cpu')
            self.num_classes = max(self.label) + 1
            self.num_features = self.data.x.size()[1]

            adj_temp = SparseTensor(row=self.data.edge_index[0], col=self.data.edge_index[1],
                                    value=torch.ones_like(self.data.edge_index[0]))
            self.adj, _ = adj_temp.saint_subgraph((class_mask == True).nonzero().view(-1))
            row, col, _ = self.adj.coo()
            self.edge_index = torch.stack([row, col], dim=0)
            self.adj = gcn_norm(self.edge_index)
            split_dataset(self)
            self.num_edges = self.data.num_edges
            self.label_train = self.label[self.train_mask]
            self.label_test = self.label[self.test_mask]
            self.label_train_onehot = onehot(self.label_train).to('cpu')




class ogbn_arxiv():
    def __init__(self, class_list=None):
        self.dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                              root='./data',
                                 transform=T.ToSparseTensor())
        self.processed_dir = self.dataset.processed_dir
        self.data = self.dataset[0]
        self.data.adj_t = self.data.adj_t.to_symmetric()
        if class_list:  # only use several classes of the entire dataset
            self.label = self.data.y.squeeze()
            self.train_mask = torch.zeros_like(self.label).bool().fill_(False)
            self.test_mask = torch.zeros_like(self.label).bool().fill_(False)
            class_mask = torch.zeros(size=(1, self.label.size()[0])).squeeze(dim=0).bool().fill_(False)
            for class_name in class_list:
                class_idx = (self.label == class_name).nonzero().view(-1)
                class_mask[class_idx] = True
            self.label = self.label[class_mask]
            self.x = self.data.x[class_mask]
            self.num_classes = max(self.label) + 1
            self.num_features = self.data.x.size()[1]
            self.data.adj_t, _ = self.data.adj_t.saint_subgraph((class_mask == True).nonzero().view(-1))
            row, col, val = self.data.adj_t.coo()
            self.edge_index = torch.stack([row, col], dim=0)
            split_idx = self.dataset.get_idx_split()
            train_idx = split_idx['train']
            test_idx = split_idx['test']
            self.train_mask[train_idx] = True
            self.test_mask[test_idx] = True
            self.train_mask = self.train_mask[class_mask]
            self.test_mask = self.test_mask[class_mask]
            self.adj = gcn_norm(self.data.adj_t)
            self.num_edges = self.data.num_edges
            self.label_train = self.label[self.train_mask].to('cpu')
            self.label_test = self.label[self.test_mask].to('cpu')
            self.label_train_onehot = onehot(self.label_train)


            self.data = None

        else:  # use all classes
            row, col, val = self.data.adj_t.coo()
            self.edge_index = torch.stack([row, col], dim=0)

            self.x = self.data.x
            self.label = self.data.y.squeeze()
            # self.label_onehot = onehot(self.label)
            self.num_classes = max(self.label) + 1
            self.num_features = self.data.x.size()[1]

            split_idx = self.dataset.get_idx_split()
            train_idx = split_idx['train']
            test_idx = split_idx['test']

            self.train_mask = torch.zeros_like(self.label).bool().fill_(False)
            self.test_mask = torch.zeros_like(self.label).bool().fill_(False)
            self.train_mask[train_idx] = True
            self.test_mask[test_idx] = True
            self.adj = gcn_norm(self.edge_index)
            self.num_edges = self.data.num_edges
            self.label_train = self.label[self.train_mask]
            self.label_test = self.label[self.test_mask]
            self.label_train_onehot = onehot(self.label_train)
            self.data = None



def split_dataset(data):
    num_train_per_class = 100
    num_test = 1000
    data.train_mask = torch.zeros(size=(1, data.label.size()[0])).squeeze(dim=0).bool().fill_(False)
    data.test_mask = torch.zeros(size=(1, data.label.size()[0])).squeeze(dim=0).bool().fill_(False)

    for c in range(data.num_classes):
        idx = (data.label == c).nonzero().view(-1)
        torch.manual_seed(seed=100)
        idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
        data.train_mask[idx] = True

    remaining = (~data.train_mask).nonzero().view(-1)
    torch.manual_seed(seed=100)
    remaining = remaining[torch.randperm(remaining.size(0))]
    data.test_mask[remaining[:num_test]] = True


def load_batch_parallel(train_loader, num_parts, num_batches_per_part, partptr, perm, current_community, multi_label=False, num_classes=None):
    num_batches = num_parts * num_batches_per_part
    i = 0
    for batch_temp in train_loader:
        if i == current_community:
            batch_temp.num_nodes = len(batch_temp.y)
            if multi_label is False:
                batch_temp.label_train = batch_temp.y[batch_temp.train_mask].to(device)
                batch_temp.label_test = batch_temp.y[batch_temp.test_mask].to(device)
                batch_temp.x = batch_temp.x.to(device)
                batch_temp.y = batch_temp.y.to(device)
                batch_temp.label_train_onehot = onehot(batch_temp.label_train, num_classes).to(device)

            batch_temp.adj_full = batch_temp.adj.to(device)
            batch_temp.perm = perm[batch_temp.node_index].to(device)
            batch_temp.edge_index = None
            temp = []
            batch_temp.nei_list = []
            for l in range(num_batches):
                node_index = torch.arange(partptr[l], partptr[l + 1])
                try:
                    adj_temp = batch_temp.adj_row.index_select(1, node_index).to(device)
                    if len(adj_temp.storage._row) > 0 and l != i:  # if is 1st-order neighbor
                        temp.append(adj_temp)
                        batch_temp.nei_list.append(l)
                except:
                    pass
            batch_temp.adj_row = None
            batch_temp.adj_list = temp
            return batch_temp
        else:
            i += 1





def load_batch(train_loader, num_parts, num_batches_per_part, partptr, perm, multi_label=False, num_classes=None):
    num_batches = num_parts * num_batches_per_part
    batch_count = 0
    batch_list = []
    for batch_temp in train_loader:
        batch_temp.num_nodes = len(batch_temp.y)
        if multi_label is False:
            batch_temp.label_train = batch_temp.y[batch_temp.train_mask].to(device)
            batch_temp.label_test = batch_temp.y[batch_temp.test_mask].to(device)
            batch_temp.label_train_onehot = onehot(batch_temp.label_train,num_classes).to(device)
        batch_temp.adj_full = batch_temp.adj.to(device)
        batch_temp.perm = perm[batch_temp.node_index].to(device)
        batch_temp.x = batch_temp.x.to(device)
        batch_temp.y = batch_temp.y.to(device)
        batch_temp.edge_index = None
        batch_list.append(batch_temp)
        batch_count += 1

    for i in range(num_parts):
        temp = []
        batch_list[i].nei_list = []
        for l in range(num_batches):
            node_index = torch.arange(partptr[l], partptr[l+1])
            try:
                adj_temp = batch_list[i].adj_row.index_select(1, node_index).to(device)
                if len(adj_temp.storage._row) > 0 and l != i: # if is 1st-order neighbor
                    temp.append(adj_temp)
                    batch_list[i].nei_list.append(l)
            except:
                pass
        batch_list[i].adj_row = None
        batch_list[i].adj_list = temp

    return batch_list

