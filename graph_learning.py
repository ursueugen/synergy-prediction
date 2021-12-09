from typing import List
import dataclasses

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

import sklearn.metrics as metrics


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
        nn.Sigmoid()
    ])
    if verbose:
        print(f"{model}")  # describe model
    return model

@dataclasses.dataclass(frozen=True, eq=True, repr=True)
class ModelMetrics:
    acc: float
    precision: float
    recall: float

def train_graph_classification(
    model: geo_nn.Sequential,
    train_loader: DataLoader, 
    test_loader: DataLoader, 
    num_epochs: int,
    verbose: bool = True):
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Utility function for accuracy
    def get_acc(model, loader) -> float:
        yhats_tensors = []
        ys_tensors = []
        for data in loader:
            outs = model(data.x, data.edge_index, data.batch)
            yhats_tensors.append(
                outs.detach()  # detach tensor from grad
            )
            ys_tensors.append(
                data.y.reshape(*outs.shape)
            )
        yhats_tensors = torch.cat(yhats_tensors, dim=0)
        yhats = np.argmax(yhats_tensors, axis=1)
        ys_tensors = torch.cat(ys_tensors, dim=0)
        ys = np.argmax(ys_tensors, axis=1)
        model_metrics = ModelMetrics(
            acc = metrics.accuracy_score(ys, yhats),
            precision = metrics.precision_score(ys, yhats, average="micro"),
            recall = metrics.recall_score(ys, yhats, average="micro")
        )
        return model_metrics
    
    for epoch in range(num_epochs):
        for data in train_loader:
            # Zero grads -> forward pass -> compute loss -> backprop
            optimizer.zero_grad()
            outs = model(data.x, data.edge_index, data.batch)
            loss = loss_fn(outs, data.y.float().reshape(outs.shape))  # no train_mask
            loss.backward()
            optimizer.step()

        # Compute accuracies
        metrics_train = get_acc(model, train_loader)
        metrics_test = get_acc(model, test_loader)
        if verbose:
            print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {loss:.4f} | "
            f"Train: {metrics_train.acc:.2f} {metrics_train.precision:.2f} {metrics_train.recall:.2f} | "
            f"Test: {metrics_test.acc:.2f} {metrics_test.precision:.2f} {metrics_test.recall:.2f}")
