'''
Graph properties to evaluate:
    
    - Number of connected components, size of top 5
    - Diameter (at least a lower bound) -> GNN width
    - Assortativity  # curiosity
    - Bridges -> to understand graph structure

    Reducing graph size
    - maximum independent set
    - dominating set
    - Ramsey (?)
    - Treewidth
    - Vertex cover
'''

import json
from datetime import datetime
import pandas as pd
import networkx as nx
import networkx.algorithms as nx_algorithms
import networkx.algorithms.clique as nx_clique

import utils


## Constants and arguments
KEEP_LARGEST_CC = True
OUTPUT_PATH = "./data/biogrid_data.json"
##


## Main
biogrid = utils.load_biogrid()
G = utils.graph_from_biogrid(biogrid, keep_largest_cc=KEEP_LARGEST_CC)

graph_data = {
    "keep_largest_cc": KEEP_LARGEST_CC,
}


### Basic stats
graph_data = dict(graph_data, **{
    "num_nodes": None,
    "num_edges": None,
    "num_connected_components": None,
    "sizes_top_connected_components": None,

})
###

### Graph diameter LB
print(f"[{datetime.now()}] [START] Graph diameter LB")
diameter_LB = nx.diameter(G) # lower bound by 2-sweep
graph_data["diameter_LB"] = diameter_LB
print(f"[{datetime.now()}] [STOP] Graph diameter LB")
###

### Assortative mixing
print(f"[{datetime.now()}] [START] Assortative mixing")
r: float = nx.degree_assortativity_coefficient(G)
graph_data["degree_assortativity"] = r
print(f"[{datetime.now()}] [STOP] Assortative mixing")
### 

### Bridges

###

### Maximum independent set
max_set: set = nx_clique.maximum_independent_set(G)
graph_data["length_max_indep_set"] = len(max_set)
###

### Minimum dominating set
min_dominating_set: set = (
    nx.algorithms.approximation
    .min_edge_dominating_set(G)
)
graph_data["length_min_dominating_set"] = len(min_dominating_set)
###

### Ramsey
###

### Treewidth
###

### Vertex cover
###

with open(OUTPUT_PATH, "w+") as fh:
    json.dump(graph_data, fh)