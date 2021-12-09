import unittest

import numpy as np

import networkx as nx
import networkx.generators.random_graphs as nx_random_graphs

import torch
import torch.nn as nn
import torch_geometric.nn as geo_nn
import torch_geometric.utils as geo_utils
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


import graph_learning


class TestGNN(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()
    
    # def test_GNN_training_loop(self):
    #     graph_learning.train_graph_classification(
    #         self.model, 
    #         self.dataloader, 
    #         self.dataloader,
    #         num_epochs=2,
    #         verbose=False)
    #     self.assertTrue(True)


class TestGNNDemo(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()
    
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

    @staticmethod
    def create_random_graph(num_nodes = 10, p = 0.3):
        return nx_random_graphs.erdos_renyi_graph(num_nodes, p)

    @staticmethod
    def create_random_graph_loader(size = 5):
        N = 10
        data_list = []
        for _ in range(size):
            g = TestGNNDemo.create_random_graph(num_nodes = N)
            data = geo_utils.from_networkx(g)
            data.x = torch.as_tensor(np.random.binomial(n = 1, p = 0.1, size = N)).reshape(N, 1).type(torch.FloatTensor)  # [N, 1]; Linear layer requires float
            data.y = torch.as_tensor(np.random.permutation([0, 1, 0])).type(torch.LongTensor)  # [3]
            data_list.append(data)
        return DataLoader(data_list, batch_size = 4)
    
    def return_TUDataset_data_example(from_loader = False):
        dataset = TUDataset(root="./data/TUDataset", name="MUTAG")
        if not from_loader:
            return dataset[0]
        else:
            loader = DataLoader(dataset, batch_size=4, shuffle=False)
            for d in loader:
                data = d
                break
            return data
    
    def test_GNN_classification_demo(self):
        model = TestGNNDemo.create_model()
        train_loader = TestGNNDemo.create_random_graph_loader(10)
        test_loader = TestGNNDemo.create_random_graph_loader(2)
        graph_learning.train_graph_classification(model, train_loader, test_loader, num_epochs = 2, verbose=False)
        self.assertTrue(True)
    


class TestGraphDataManipulation(unittest.TestCase):

    @staticmethod
    def build_test_graph(
        attribute_name: str, 
        attribute_values: list
        ) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(["a", "b", "c"])
        G.add_edges_from([["a", "b"], ["b", "c"], ["a", "c"]])
        for i, node in enumerate(G.nodes):
                G.nodes[node][attribute_name] = attribute_values[i]
        return G

    def setUp(self) -> None:
        self.ATTRIBUTE_NAME = "TEST_ATTRIBUTE"
        self.ATTRIBUTE_VALUES = [1, 2, 3]
        self.G: nx.Graph = TestGraphDataManipulation.build_test_graph(self.ATTRIBUTE_NAME, self.ATTRIBUTE_VALUES)
        return super().setUp()
    
    def test_node_attributes_addition(self):
        self.assertEqual(len(dict(self.G.nodes.data())), len(self.G.nodes))
    
    def test_node_attributes_to_tensor(self):
        data: Data = geo_utils.from_networkx(
            G = self.G, 
            group_node_attrs = [self.ATTRIBUTE_NAME]
        )
        manual_node_features_tensor = torch.Tensor([self.ATTRIBUTE_VALUES]).reshape(3, 1)

        self.assertTrue( all(torch.eq(data.x, manual_node_features_tensor)) )


if __name__ == "__main__":
    unittest.main()