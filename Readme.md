# Predicting synergism from PPI network.

## Usage

Install the `conda` environment `env.yml`.

## Datasets

- synergyage_0: dataset with partial recovery of gene names and interventions encoded as node indicators on the graph (not graph intervention).
- synergyage_0_10samples: 10 samples from the above dataset
- synergyage_1: Dataset with reduced graph size - only the synergyage genes and their 1-neighbors. Doesn't help much because the subgraph is still large - 6893 nodes.
- synergyage_1.1: Dataset with reduced graph size. Same strategy as above, only applied on a small random subset of SynergyAge genes: ['ced-4', 'par-5', 'sqt-1', 'daf-2', 'nuc-1', 'skn-1', 'mes-4', 'daf-18', 'nuo-5', 'cmk-1']  # TODO

## Planning

1. Setup the system and perform a prediction - Done \[about 52% (multi-class) accuracy\]

- Deal with the gene names issue 
- Add monitoring system for models and setup infrastructure.
- Perform predictions on relevant subgraphs (deal with sparsity)
- Improve performance: multi-cpus, gpus

2. Perform a similar naive prediction on new interventions, not present in the training set - harder problem.
- gene interventions bias (duplicates)

3. Perform the prediction on graph structure rather than node features, i.e. graph interventions.

## Ideas

- Preprocess the graph to ameliorate sparsity

- Unsupervised learning for the dataset graphs