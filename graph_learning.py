from typing import List, OrderedDict
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


def loader_from_datalist(datalist: List[Data], batch_size: int = 1) -> DataLoader:
    return DataLoader(datalist, batch_size=batch_size)


def create_GNN_model(
    input_dim, hidden_dim, output_dim, verbose: bool = False
) -> geo_nn.Sequential:
    model = geo_nn.Sequential(
        "x, edge_index, batch",
        [
            (geo_nn.GCNConv(input_dim, hidden_dim), "x, edge_index -> x"),
            nn.ReLU(inplace=True),
            (geo_nn.GCNConv(hidden_dim, hidden_dim), "x, edge_index -> x"),
            nn.ReLU(inplace=True),
            (geo_nn.global_mean_pool, "x, batch -> x"),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        ],
    )
    if verbose:
        print(f"{model}")  # describe model
    return model


def create_GAT_model(
    input_dim, hidden_dim, output_dim, verbose: bool = False
) -> geo_nn.Sequential:
    model = geo_nn.Sequential(
        "x, edge_index, batch",
        [
            (geo_nn.GATv2Conv(input_dim, hidden_dim), "x, edge_index -> x"),
            nn.ReLU(inplace=True),
            (geo_nn.GATv2Conv(hidden_dim, hidden_dim), "x, edge_index -> x"),
            nn.ReLU(inplace=True),
            (geo_nn.global_mean_pool, "x, batch -> x"),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        ],
    )
    return model


class GraphNN(torch.nn.module):
    def __init__(
        self, 
        input_dim: int = 1, 
        hidden_dim: int = 16, 
        output_dim: int = 3,
        num_latent_layers: int = 1):
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.GCN_in_to_latent = geo_nn.GCNConv(
            self.input_dim, self.hidden_dim)
        self.GCN_in_to_latent_inputs = "x, edge_index -> x"

        self.GCN_latent_to_latent = geo_nn.GCNConv(
            self.hidden_dim, self.hidden_dim)
        self.GCN_latent_to_latent_inputs = "x, edge_index -> x"

        self.activation = nn.ReLu(inplace=True)

        self.graph_global_pool = geo_nn.global_mean_pool
        self.graph_global_pool_inputs = "x, batch -> x"

        self.classifier = nn.Sequential(
            OrderedDict(
                "Linear", nn.Linear(self.hidden_dim, self.output_dim), 
                "Sigmoid", nn.Sigmoid(),
            )
        )

        self.model = geo_nn.Sequential(
            "x, edge_index, batch",
            [
                (self.GCN_in_to_latent, self.GCN_in_to_latent_inputs),
                self.activation,
                *[
                    (
                        (self.GCN_latent_to_latent, self.GCN_latent_to_latent_inputs),
                        self.activation,
                    ) for _ in num_latent_layers
                ],
                (self.graph_global_pool, self.graph_global_pool_inputs),
                self.classifier,
            ],
        )
        super().__init__()

    def forward(self, data):
        return self.model(data)


@dataclasses.dataclass(frozen=True, eq=True, repr=True)
class ModelMetrics:
    acc: float
    precision: float
    recall: float
    confusion_matrix: np.array


def train_graph_classification(
    model: geo_nn.Sequential,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    weighted_loss: bool = False,
    learning_rate: float = 0.01,
    verbose: bool = True,
):

    if not weighted_loss:
        loss_fn = nn.CrossEntropyLoss()
    else:
        class_array = np.array(
            list(map(lambda data: data.y.numpy(), train_loader.dataset))
        )
        weights = torch.Tensor(class_array.sum(axis=0) / class_array.shape[0])
        loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4,)

    # Utility function for accuracy
    def get_acc(model, loader) -> float:
        yhats_tensors = []
        ys_tensors = []
        for data in loader:
            outs = model(data.x, data.edge_index, data.batch)
            yhats_tensors.append(outs.detach())  # detach tensor from grad
            ys_tensors.append(data.y.reshape(*outs.shape))
        yhats_tensors = torch.cat(yhats_tensors, dim=0)
        yhats = np.argmax(yhats_tensors, axis=1)
        ys_tensors = torch.cat(ys_tensors, dim=0)
        ys = np.argmax(ys_tensors, axis=1)
        model_metrics = ModelMetrics(
            acc=metrics.accuracy_score(ys, yhats),
            precision=metrics.precision_score(ys, yhats, average="micro"),
            recall=metrics.recall_score(ys, yhats, average="micro"),
            confusion_matrix=metrics.confusion_matrix(ys, yhats),
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
            print(
                f"[Epoch {epoch+1}/{num_epochs}] Loss: {loss:.4f} | "
                f"Train: {metrics_train.acc:.2f} {metrics_train.precision:.2f} {metrics_train.recall:.2f} | "
                f"Test: {metrics_test.acc:.2f} {metrics_test.precision:.2f} {metrics_test.recall:.2f} |\n"
                f"Confusion_matrix: \n{metrics_train.confusion_matrix}\n"
            )
