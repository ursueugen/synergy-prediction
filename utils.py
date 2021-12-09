from typing import Set
import networkx as nx
import pandas as pd


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


def graph_from_biogrid(df_biogrid: pd.DataFrame) -> nx.Graph:
    return nx.from_pandas_edgelist(
        df_biogrid[[GENE_NAME_A_COL, GENE_NAME_B_COL]], 
        GENE_NAME_A_COL, 
        GENE_NAME_B_COL
    )