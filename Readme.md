# Predicting synergism from PPI network.

## Usage

Install the `conda` environment `env.yml`.

## Datasets

- synergyage_0: dataset with partial recovery of gene names and interventions encoded as node indicators on the graph (not graph intervention).
- synergyage_0_10samples: 10 samples from the above dataset
- synergyage_1: Dataset with reduced graph size - only the synergyage genes and their 1-neighbors. Doesn't help much because the subgraph is still large - 6893 nodes.
- synergyage_1.1: [might not be full atm, but workable] Dataset with reduced graph size. Same strategy as above, only applied on a small random subset of SynergyAge genes: ['ced-4', 'par-5', 'sqt-1', 'nuc-1', 'skn-1', 'mes-4', 'daf-18', 'nuo-5', 'cmk-1']  # 'daf-2' removed, too connected.

Todo:
- synergyage_2: Dataset with modified graph structure based on the interventions.  # TODO
- dataset with distinguishing neighboring nodes through a class label  # ameliorate sparsity

## Planning

0. Analyze the C. elegans PPI network for statistics that could be helpful in building the optimal GNN architecture. For example, the diameter of the graph might suggest the depth of the GNN. An advantage is that all the predictions occur on the basic graph structure and, therefore, there is a way to uniquely assign ids to the nodes for every input. Evaluate the minimal vertex cover, the maximal independent set.

1. Setup the system and perform a prediction

- Deal with the gene names issue
- Add monitoring system for models and setup infrastructure.
- Perform predictions on relevant subgraphs (deal with sparsity)
- Improve performance: multi-cpus, gpus

2. Perform a similar naive prediction on new interventions, not present in the training set - harder problem.
- gene interventions bias (duplicates)

3. Perform the prediction on graph structure rather than node features, i.e. graph interventions.


## Questions

- What is the graph generation model for the PPI? What  properties can be inferred from this?


## Ideas

- Graph reduction by coarsening. Implement separately or insert coarsening layers? What about graph partitioning?
    - Interpretation via contraction sets
    - Coarse graph should capture the global patterns, while details can be recovered in refinements.
    - local pooling might work as well in reducing size
        - max_local according to a "clustering", so need to define ~contraction sets anyways?
    - **Try first: Algebraic JC: fast and competitive.**
    - 

- **Break the curse of anonymouse nodes**: https://andreasloukas.blog/2019/12/27/what-gnn-can-and-cannot-learn/
    - node degree could be an improvement
    - **interpolate the intervention signal across the neighbors** --> could be useful for sparse signals on nodes  # this should be done by the GNN, maybe do it as desperate effor in case other solutions fail.

- Preprocess the graph to ameliorate sparsity

- Unsupervised learning for the dataset graphs

## Useful references

### Loukas 2018: Graph reduction with spectral and cut guarantees.

### Huang 2021: Scaling up graph neural networks via graph coarsening.
- simply applying off-the-shelf coarsening methods, we can reduce #nodes <10x withough noticable reduction in classification accuracy.
- key property of the Laplacian L: quad form measures smoothnes of a signal w.r.t. the graph structure and thus often used for regularization purposes.
- check section 2.2 for smoothness and graph FT
- Reffer to APPNP (instead of GCN): uses propagation rules inspired from personalized PageRank
- Graph coarsening can be efficiently pre-computed on CPUs, where main memory size could be much larger than GPUs.
- Test classification performance of four coarsening methods discussed in \[27 - Loukas 2018\]: Variation Neighborhoods, Variation Edges, Algebraic JC, A

## Some conclusions on C.elegans PPI

- The graph is disconnected, but there is one huge connected component (CC)
- 