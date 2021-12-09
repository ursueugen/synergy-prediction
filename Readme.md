# Predicting synergism from PPI network.

## Usage

Install the `conda` environment `env.yml`.

## Datasets

- synergyage_0: dataset with partial recovery of gene names and interventions encoded as node indicators on the graph (not graph intervention).
- synergyage_0_10samples: 10 samples from the above dataset

## Planning

1. Setup the system and perform a prediction - Done \[about 52% (multi-class) accuracy\]

- Deal with the gene names issue 
- Add monitoring system for models and setup infrastructure.

2. Perform a similar naive prediction on new interventions, not present in the training set - harder problem.
- check class imbalance (and appropriate metrics)
- gene interventions bias (duplicates)

3. Perform the prediction on graph structure rather than node features, i.e. graph interventions.

## Ideas

- Reflect uncertainty of effect of some combinations through encoding class probabilities.

- Unsupervised learning for the dataset graphs