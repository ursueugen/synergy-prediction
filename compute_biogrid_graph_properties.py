'''
Graph properties to evaluate:
    
    - Number of connected components, size of top 5
    - Diameter (at least a lower bound) -> GNN width
    - Assortativity  # curiosity
    - Bridges -> to understand graph structure
    - Communicability -> ?

    Reducing graph size
    - maximum independent set
    - dominating set
    - Ramsey (?) - largest clique and maximum independent set
    - Treewidth
    - Vertex cover
    - Communities [networkx]
'''

import json
import functools
from typing import Callable
import warnings
from datetime import datetime
import pandas as pd
import networkx as nx
import networkx.algorithms as nx_algorithms

import utils


KEEP_LARGEST_CC = True
OUTPUT_PATH = "./data/biogrid_data.json" 


def print_execution_time(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[{datetime.now()}] [START] {func.__name__}")
        result = func(*args, **kwargs)
        print(f"[{datetime.now()}] [FINISH] {func.__name__}")
        return result
    return wrapper

@print_execution_time
def diameter_lower_bound(G: nx.Graph) -> float:
    return nx.diameter(G)

@print_execution_time
def assortativity(G: nx.Graph) -> float:
    return nx.degree_assortativity_coefficient(G)

@print_execution_time
def num_bridges(G: nx.Graph) -> int:
    bridges = list(nx_algorithms.bridges(G))
    return len( bridges )

@print_execution_time
def max_clique_len(G: nx.Graph) -> int:
    max_clique: set = nx_algorithms.approximation.max_clique(G)
    return len(max_clique)

@print_execution_time
def max_indep_set_size(G: nx.Graph) -> int:
    max_set: set = nx_algorithms.approximation.maximum_independent_set(G)
    return len(max_set)

@print_execution_time
def min_dominating_set_size(G: nx.Graph) -> int:
    min_dominating_set: set = (
        nx_algorithms.approximation
        .min_edge_dominating_set(G)
    )
    return len(min_dominating_set)

@print_execution_time
def treewidth(G: nx.Graph) -> int:
    treewidth, _ = nx_algorithms.approximation.treewidth_min_degree(G)
    return treewidth

@print_execution_time
def min_weighted_vertex_cover_size(G: nx.Graph) -> int:
    min_weighted_cover: set = (
        nx_algorithms.approximation
        .min_weighted_vertex_cover(G)
    )
    return len(min_weighted_cover)


if __name__ == "__main__":

    biogrid = utils.load_biogrid()
    G = utils.graph_from_biogrid(biogrid, keep_largest_cc=KEEP_LARGEST_CC)

    graph_data = {
        "keep_largest_cc": KEEP_LARGEST_CC,
        "warnings": [
            "Communicability not computed.",
            "Modularity not computed",
        ]
    }

    graph_data = dict(graph_data, **{
        "num_nodes": len(G.nodes),
        "num_edges": len(G.edges),
        "num_connected_components": nx_algorithms.components.number_connected_components(G),
    })

    graph_data["diameter_LB"] = diameter_lower_bound(G)
    graph_data["degree_assortativity"] = assortativity(G)
    graph_data["num_bridges"] = num_bridges(G)
    graph_data["max_clique_len"] = max_clique_len(G)
    graph_data["max_indep_set_size"] = max_indep_set_size(G)
    graph_data["min_dominating_set_size"] = min_dominating_set_size(G)
    graph_data["treewidth"] = treewidth(G)
    graph_data["min_weighted_vertex_cover_size"] = min_weighted_vertex_cover_size(G)

    with open(OUTPUT_PATH, "w+") as fh:
        json.dump(graph_data, fh)