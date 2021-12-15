from typing import Set, Tuple, List, Union
from pathlib import Path
import numpy as np
import networkx as nx
import pandas as pd
import torch
import torch_geometric.utils as geo_utils
from torch_geometric.data import Data




INTERVENTION_CLASSES = ["NS", "PRO", "ANTI"]
GENE_NAME_A_COL = "Official Symbol Interactor A"
GENE_NAME_B_COL = "Official Symbol Interactor B"
DATALISTS_SAVE_DIR = "./data/datalists"


def load_biogrid() -> pd.DataFrame:
    return pd.read_csv("./data/BIOGRID-ORGANISM-Caenorhabditis_elegans-4.4.203.tab3.txt", sep='\t')

def get_biogrid_gene_names(df_biogrid: pd.DataFrame) -> Set[str]:
    return set(df_biogrid[GENE_NAME_A_COL]).union(df_biogrid[GENE_NAME_B_COL])


def load_synergyage() -> pd.DataFrame:
    return pd.read_csv("./data/synergyage_database_augmented.tsv", sep='\t')

def get_synergyage_genes(df_synergy: pd.DataFrame) -> Set[str]:
    genes_syn = set([])
    for genes_str in df_synergy["genes_str"]:
        genes_syn = genes_syn.union( set( genes_str.split(";") ) )
    return genes_syn

def get_synergyage_subset_with_biogrid_gene_names() -> pd.DataFrame:
    """
    Return: SynergyAge dataframe subset with genes whose names
     are readily found in BioGrid.
    """
    df_biogrid = load_biogrid()
    biogrid_genes = get_biogrid_gene_names(df_biogrid)
    
    df_syn = load_synergyage()
    selected = []
    for i, row in df_syn.iterrows():
        genes = row["genes_str"].split(";")
        row_has_gene_not_in_biogrid = False
        for gene in genes:
            if gene not in biogrid_genes:
                row_has_gene_not_in_biogrid = True
                break
        if row_has_gene_not_in_biogrid:
            continue
        else:
            selected.append(i)
    df_syn = df_syn.iloc[selected, :].copy()
    df_syn = df_syn.iloc[np.random.permutation(len(df_syn)).tolist()].copy()
    return df_syn


def encode_intervention_class(intervention_class: str) -> torch.Tensor:
    """
    Args:
     - intervention_class: the class of the SynergyAge intervention in {NS, PRO, ANTI}
    Return: torch tensor representing the class as an indicator vector.
    """
    if intervention_class == "NS":
        code = [0, 1, 0]
    elif intervention_class == "PRO":
        code = [0, 0, 1]
    elif intervention_class == "ANTI":
        code = [1, 0 ,0]
    else:
        raise ValueError(f"Intervention class {intervention_class} not recognized.")
    return torch.as_tensor(code).type(torch.LongTensor)


def build_datalist(
    synergy: pd.DataFrame, 
    biogrid: pd.DataFrame,
    subgraph: bool = False,
    reduced_synergyage: bool = False,
    ) -> Tuple[list, list]:
    
    G = graph_from_biogrid(biogrid)

    if reduced_synergyage:
        synergy = select_synergyage_genes(
            synergy,
            ['ced-4', 'par-5', 'sqt-1', 'nuc-1', 'skn-1', 'mes-4', 'daf-18', 'nuo-5', 'cmk-1']
        )

    if subgraph:
        synergyage_genes = get_synergyage_genes(synergy)
        
        # Get 1-neighbors of intervention genes
        SG_nodes = set(synergyage_genes)
        for gene in synergyage_genes:
            for n in G.neighbors(gene):
                SG_nodes.add(n)
        
        SG = G.__class__()
        edge_list = []
        for n, nbrs in G.adj.items():
            if n not in SG_nodes:
                continue
            for nbr in set(nbrs.keys()):
                edge_list.append((n, nbr))
        SG.add_edges_from(edge_list)
        G = SG.copy()
        del SG

    data_list = []
    for i, row in synergy.iterrows():
        intervention_genes = row["genes_str"].split(";")
        # For removing the nodes that are KO
        # biogrid_mask = ( (~df_biogrid[GENE_NAME_A_COL].isin(intervention_genes)) & (~df_biogrid[GENE_NAME_B_COL].isin(intervention_genes)))
        # df_biogrid_trimmed = df_biogrid.loc[biogrid_mask].copy()

        G_intervention = G.copy()

        for node in G_intervention.nodes:
            if node in intervention_genes:
                G_intervention.nodes[node]["HAS_INTERVENTION"] = 1
            else:
                G_intervention.nodes[node]["HAS_INTERVENTION"] = 0
        # graph_list.append(G)

        class_encoded: torch.Tensor = encode_intervention_class(row["LIFESPAN_CLASS"])
        # target_list.append(class_encoded)
    
        data = geo_utils.from_networkx(G_intervention, group_node_attrs=["HAS_INTERVENTION"])
        data.intervention_genes = intervention_genes
        data.source_idx = i
        data.x = data.x.type(torch.FloatTensor)
        data.y = class_encoded
        assert data.y.type() == 'torch.LongTensor'
        data_list.append(data)
    return data_list

def select_synergyage_genes(
    df_syn: pd.DataFrame, 
    genes_selection: List[str]
    ) -> pd.DataFrame:
    rows_selected = []
    for i, row in df_syn.iterrows():
        intervention_genes = row["genes_str"].split(";")
        intersection = set(intervention_genes).intersection(set(genes_selection))
        if len(intersection) > 0:
            rows_selected.append(row)
    df_syn_selection = pd.concat(rows_selected, axis=1).transpose()
    return df_syn_selection


def save_datalist(datalist: List[Data], name: str, override: bool = False) -> None:
    dir_path = Path(".") / DATALISTS_SAVE_DIR / name
    dir_path.mkdir(parents=True, exist_ok=True)
    if dir_path.exists() and (not override):
        raise RuntimeError
    for i, data in enumerate(datalist):
        fpath = dir_path / f"{name}-{i}.pt"
        torch.save(data, fpath)

def load_datalist(dir_path: Union[Path, str]) -> List[Data]:
    if not Path(dir_path).exists():
        raise RuntimeError(f"Directory doesn't exist: {dir_path}")
    else:
        data_list = []
        for fpath in Path(dir_path).glob("*.pt"):
            data = torch.load(fpath)
            data_list.append(data)
        return data_list


def graph_from_biogrid(
    df_biogrid: pd.DataFrame, 
    keep_largest_cc: bool = False
    ) -> nx.Graph:

    G = nx.from_pandas_edgelist(
        df_biogrid[[GENE_NAME_A_COL, GENE_NAME_B_COL]], 
        GENE_NAME_A_COL, 
        GENE_NAME_B_COL
    )

    ccs = list(nx.connected_components(G))
    ccs_len = list(map(len, ccs))
    ccs_len_max = max(ccs_len)
    max_len_cc_idx = ccs_len.index(ccs_len_max)
    largest_cc = ccs[max_len_cc_idx]
    return G.subgraph(largest_cc)