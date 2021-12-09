# Predicting synergism from PPI network.

## Usage

Install the `conda` environment `env.yml`.

## Planning

1. Setup the system and perform a prediction - Done \[about 52% (multi-class) accuracy\]

1a. Analyze the dataset for:

- class imbalance (add appropriate metrics)  # Cross-entropy weights?
- gene interventions bias (duplicates)

1b. Adjust the learning objective to learning probabilities
- Cross-entropy loss

1c. Add monitoring system for models and setup infrastructure.

2. Perform a similar naive prediction on new interventions, not present in the training set - harder problem.

3. Perform the prediction on graph structure rather than node features, i.e. graph interventions.

## Ideas

- Reflect uncertainty of effect of some combinations through encoding class probabilities.

- Unsupervised learning for the dataset graphs