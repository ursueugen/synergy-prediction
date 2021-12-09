from typing import Set, Tuple
import numpy as np
import networkx as nx
import pandas as pd
import torch
import torch_geometric.utils as geo_utils



INTERVENTION_CLASSES = ["NS", "PRO", "ANTI"]
GENE_NAME_A_COL = "Official Symbol Interactor A"
GENE_NAME_B_COL = "Official Symbol Interactor B"


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

def get_synergyage_subset_with_biogrid_gene_names() -> pd.Dataframe:
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


def build_graph_and_target_lists(
    synergy: pd.DataFrame, 
    biogrid: pd.DataFrame
    ) -> Tuple[list, list]:
    
    graph_list = []
    target_list = []
    for _, row in synergy.iterrows():
        intervention_genes = row["genes_str"].split(";")
        # For removing the nodes that are KO
        # biogrid_mask = ( (~df_biogrid[GENE_NAME_A_COL].isin(intervention_genes)) & (~df_biogrid[GENE_NAME_B_COL].isin(intervention_genes)))
        # df_biogrid_trimmed = df_biogrid.loc[biogrid_mask].copy()

        G = nx.from_pandas_edgelist(
            biogrid[[GENE_NAME_A_COL, GENE_NAME_B_COL]],
            source=GENE_NAME_A_COL,
            target=GENE_NAME_B_COL,
        )
        for node in G.nodes:
            if node in intervention_genes:
                G.nodes[node]["HAS_INTERVENTION"] = 1
            else:
                G.nodes[node]["HAS_INTERVENTION"] = 0
        graph_list.append(G)

        class_encoded: torch.Tensor = encode_intervention_class(row["LIFESPAN_CLASS"])
        target_list.append(class_encoded)
    return (graph_list, target_list)


def build_datalist(graph_list: list, target_list: list) -> list:
    data_list = []
    for graph, target in zip(graph_list, target_list):
        data = geo_utils.from_networkx(graph, group_node_attrs=["HAS_INTERVENTION"])
        data.x = data.x.type(torch.FloatTensor)
        data.y = target
        assert data.y.type() == 'torch.LongTensor'
        data_list.append(data)
    return data_list


def graph_from_biogrid(df_biogrid: pd.DataFrame) -> nx.Graph:
    return nx.from_pandas_edgelist(
        df_biogrid[[GENE_NAME_A_COL, GENE_NAME_B_COL]], 
        GENE_NAME_A_COL, 
        GENE_NAME_B_COL
    )