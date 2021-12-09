"""
Create datasets for graph learning.
"""

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric.utils as geo_utils
from torch_geometric.loader import DataLoader

from utils import *


df_syn = load_synergyage()
df_biogrid = load_biogrid()


def encode_intervention_class(intervention_class: str) -> torch.Tensor:        
    if intervention_class == "NS":
        code = [0, 1, 0]
    elif intervention_class == "PRO":
        code = [0, 0, 1]
    elif intervention_class == "ANTI":
        code = [1, 0 ,0]
    else:
        raise ValueError(f"Intervention class {intervention_class} not recognized.")
    return torch.as_tensor(code).type(torch.LongTensor)