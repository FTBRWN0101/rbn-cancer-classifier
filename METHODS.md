# Methods

This file sets out what each stage of the pipeline does, with its parameters, then the caveats that affect how to read the results.
Names in backticks are the matching identifiers in the code.

## Data preparation

`src/data/` turns the 253 spectra into binary profiles.

1. `load_ovarian_data` reads `data/raw_data/Ovarian.arff`: 253 samples by 15,154 M/Z intensities.
   It codes the labels in alphabetical order, so Cancer is class 0 and Normal is class 1.
2. `DataPreprocessor` fills missing values with column means and drops constant features, though the file has neither.
   It then min-max scales each feature to [0, 1].
3. `DataSplitter` makes one stratified 70/30 split with seed 42.
   The training set holds 176 samples (113 Cancer, 63 Normal) and the test set 77 (49 Cancer, 28 Normal).
4. `MedianDiscretizer` sets each feature to 1 where its value is at or above that feature's training median, and to 0 otherwise.

## Feature selection

`select_features_with_threshold` keeps the features whose class means differ by at least 0.1 on the training set: 9,024 of the 15,154.
It ranks them by three scores: mutual information with the label, Fisher score and the absolute difference in class means.
`combined_importance` averages each feature's rank across the three scores, and the best 30 go forward.

## Random Boolean network

`DataDrivenRBN` has N = 20 nodes.
Each node reads K = 3 other nodes, chosen at random with seed 42.
Its update rule is a weighted threshold learned from the training set:

```math
x_i(t+1) = \left[\, \sum_{j=1}^{K} w_{ij}\, x_{c_{ij}}(t) + b_i > 0 \,\right]
```

Here c<sub>ij</sub> is node i's j-th input, w<sub>ij</sub> is the Pearson correlation between that input and node i across the training samples, and b<sub>i</sub> is node i's mean training activity minus 0.5.
All nodes update at once.

`RBNSimulator` starts the network from a sample's 30 binary features and records 200 states.
`AttractorAnalyzer` finds the first state that repeats.
The states from its first visit up to the repeat form the attractor, and the states before it form the transient.

## Trajectory features

`CombinedFeatureExtractor` turns each trajectory into 12 features:

| Feature | What it measures |
| --- | --- |
| `attractor_length` | Number of states in the attractor cycle |
| `mean_activity` | Fraction of nodes on, over the attractor |
| `std_activity` | Each node's standard deviation over the attractor, averaged over nodes |
| `entropy` | Each node's Shannon entropy over the attractor, averaged over nodes |
| `hamming_diversity` | Mean Hamming distance between consecutive attractor states |
| `frozen_nodes` | Fraction of nodes that stay constant over the attractor |
| `complexity` | Distinct states divided by attractor length |
| `energy` | Mean number of nodes on, over the attractor |
| `transient_length` | Number of steps before the attractor |
| `transient_activity` | Fraction of nodes on, over the transient |
| `settling_stability` | Mean Hamming distance between consecutive transient states |
| `transient_entropy` | Each node's Shannon entropy over the transient, averaged over nodes |

## Classification

`RBNAttractorClassifier` standardises the 12 features and fits a logistic regression with balanced class weights (C = 1, seed 42).
`optimize_threshold` then sets the decision threshold on the training set.
It tries 50 values from 0.2 to 0.8 and keeps the one that maximises F1 for class 1, Normal.
In the original run it chose 0.408.
The RBN feature-importance output reports the absolute logistic-regression coefficients.

The baseline is a random forest of 100 trees, with a maximum depth of 10, balanced class weights and seed 42.
It trains on the 30 binary features directly.
The code scores both models once, on the 77 test samples.

## Noise test

`NoiseRobustnessEvaluator` flips each binary test input with probability ρ ∈ {0, 0.05, 0.1, 0.2} and rescores the trained RBN classifier.

## NK landscape

`src/nk_landscape/` is a demo that is separate from the classifier.
It builds an NK landscape (N = 20, K = 2, seed 42) and runs simulated annealing for 1,000 steps, from a temperature of 1 cooled by a factor of 0.995 per step.
It then reports the lag-1 autocorrelation of fitness over 100 random genomes.
Those genomes are independent, not steps of a walk, so this autocorrelation is near zero by construction and does not measure ruggedness.

## Caveats

- **Tied features.** Thousands of binary features tie exactly on all three selection scores, so numpy's sort order picks the last few of the 30.
  That order can vary between machines, and so can the results.
- **Scaling before the split.** `DataPreprocessor` fits its min-max scaling on all 253 samples, not on the training set alone.
  Kent Ridge had already scaled the whole dataset to [0, 1], with one minimum and maximum over every sample and feature.
  Binarisation only compares a value with its training median, so fitting the scaling on the training set alone changes 14 of the 1.17 million test values, none of them in the selected 30.
- **Network size.** The network has 20 nodes but receives 30 features.
  `step` updates the first 20 positions and sets the other 10 to 0, so features 21 to 30 only set the starting state.
- **Redundant features.** `complexity` is always 1, because attractor states are distinct by construction.
  `energy` is always 30 times `mean_activity`.
  The 12 features therefore carry 10 separate signals.
- **One split.** `optimize_threshold` tunes the decision threshold on the training set, and there is no validation set.
  Every reported number rests on one set of 77 test samples.
