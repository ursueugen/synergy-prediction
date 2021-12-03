import numpy as np
import networkx as nx
import networkx.generators.random_graphs as nx_random_graphs

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.nn as geo_nn
import torch_geometric.utils as geo_utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graph_learning import train_graph_classification 

def create_model():
    INPUT_DIM: int = 1
    HIDDEN_DIM: int = 32
    OUTPUT_DIM: int = 3
    model = geo_nn.Sequential('x, edge_index, batch', [
        (geo_nn.GCNConv(INPUT_DIM, HIDDEN_DIM), 'x, edge_index -> x'),
        nn.ReLU(inplace = True),
        (geo_nn.GCNConv(HIDDEN_DIM, HIDDEN_DIM), 'x, edge_index -> x'),
        nn.ReLU(inplace = True),
        (geo_nn.global_mean_pool, 'x, batch -> x'),
        nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
    ])
    return model

def create_toy_graph():
    g = nx.Graph()
    g.add_nodes_from([
        "A", 
        "B", 
        "C"
    ])
    g.add_edges_from([("A", "B"), ("B", "C")])
    return g

def create_random_graph(num_nodes = 10, p = 0.3):
    return nx_random_graphs.erdos_renyi_graph(num_nodes, p)

def create_random_graph_loader(size = 5):
    N = 10
    data_list = []
    for _ in range(size):
        g = create_random_graph(num_nodes = N)
        data = geo_utils.from_networkx(g)
        data.x = torch.as_tensor(np.random.binomial(n = 1, p = 0.1, size = N)).reshape(N, 1).type(torch.FloatTensor)  # [N, 1]; Linear layer requires float
        data.y = torch.as_tensor(np.random.permutation([0, 1, 0])).type(torch.LongTensor)  # [3]
        data_list.append(data)
    return DataLoader(data_list, batch_size = 4)

def return_TUDataset_data_example(from_loader = False):
    dataset = TUDataset(root="./TUDataset", name="MUTAG")
    if not from_loader:
        return dataset[0]
    else:
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        for d in loader:
            data = d
            break
        return data



if __name__ == "__main__":

    model = create_model()
    train_loader = create_random_graph_loader(100)
    test_loader = create_random_graph_loader(20)
    print("Start training")
    train_graph_classification(model, train_loader, test_loader, num_epochs = 30)

    # Likely issue: tensor dimensions for batched inputs

    # train_loader = DataLoader(train_data_list, batch_size = 64, shuffle = True)
    # test_loader = DataLoader(test_data_list, batch_size = 64, shuffle = False)