from typing import List
# import dataclasses

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


# def datalist_from_graph_list(
#     graph_list: List[nx.Graph], 
#     batch_size: int
#     ) -> List[Data]:
#     data_list = []
#     for g in graph_list:
#         data = geo_utils.from_networkx(g)
#         data.x = None  # extract from networkx somehow :)
#         raise NotImplementedError
#         data.x = data.x.reshape(data.x.shape[0], 1)  # [Num_nodes, num_features]
#         data.x = data.x.type(torch.FloatTensor)  # Linear layer requires float
#         data.y = torch.as_tensor(np.random.permutation([0, 1, 0])).type(torch.LongTensor)  # [3]
#         data_list.append(data)
#     return data_list

def loader_from_datalist(
    datalist: List[Data], 
    batch_size: int = 1
    ) -> DataLoader:
    return DataLoader(datalist, batch_size = batch_size)

def create_GNN_model(
    input_dim, 
    hidden_dim, 
    output_dim, 
    verbose: bool = False
    ) -> geo_nn.Sequential:
    model = geo_nn.Sequential('x, edge_index, batch', [
        (geo_nn.GCNConv(input_dim, hidden_dim), 'x, edge_index -> x'),
        nn.ReLU(inplace = True),
        (geo_nn.GCNConv(hidden_dim, hidden_dim), 'x, edge_index -> x'),
        nn.ReLU(inplace = True),
        (geo_nn.global_mean_pool, 'x, batch -> x'),
        nn.Linear(hidden_dim, output_dim),
    ])
    if verbose:
        print(f"{model}")  # describe model
    return model


def train_graph_classification(
    model: geo_nn.Sequential,
    train_loader: DataLoader, 
    test_loader: DataLoader, 
    num_epochs: int,
    verbose: bool = True):
    
    num_classes = train_loader.dataset[0].y.shape[0]
    # batch_size = train_loader.batch_size
    
    loss_fn = nn.BCEWithLogitsLoss()  # Use cross-entropy + soft-max to get probability output
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Utility function for accuracy
    def get_acc(model, loader) -> float:
        n_total = 0
        n_ok = 0
        for data in loader:
            outs = model(data.x, data.edge_index, data.batch).reshape(data.y.shape[0])
            n_ok += ((outs>0) == data.y).sum().item()
            n_total += data.y.shape[0]
        return n_ok / n_total
    
    for epoch in range(num_epochs):
        for data in train_loader:
            # Zero grads -> forward pass -> compute loss -> backprop
            optimizer.zero_grad()
            outs = model(data.x, data.edge_index, data.batch).reshape(data.y.shape[0])
            loss = loss_fn(outs, data.y.float())  # no train_mask
            loss.backward()
            optimizer.step()

        # Compute accuracies
        acc_train = get_acc(model, train_loader)
        acc_test = get_acc(model, test_loader)
        if verbose:
            print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {loss} | Train: {acc_train:.3f} | Test: {acc_test:.3f}")
