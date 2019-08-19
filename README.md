# Introduction

![PyPI - Downloads](https://img.shields.io/pypi/dm/genetics?color=lightgreen&label=PyPI)

Genetic Algorithm in Python, which could be used for Sampling, Feature Select, Model Select, etc in Machine Learning


# Install

```shell

sudo pip3 install genetics

```

# Usage

## API

### Initialize the Object

```python
GA(train, valid, estimator, groups=100, iter=200, r_sample=0.8, r_crossover=0.5, r_vary=0.01, r_keep_best=0.1, n_jobs=4, verbose=False)
```

- train: a 2D numpy matrix, the last column will be used as the labels
- valid: a 2D numpy matrix, which's columns should be the same as the train
- estimator: a SKLearn estimator, such as RandomForestClassifier or SVR etc.
- groups: the groups in every generation, default 200
- iter: the number of iterations, the procedure will stop when reach the number. default 200
- r_sample: useful when doing sampling, the ratio for sampling, deault 0.8
- r_crossover: the ratio of crossover when generating a children from his parents, default 0.5
- r_vary: the ratio for varying when generating a child from his parents, default 0.01, suggest 0 - 0.1
- r_keep_best: the ratio for keeping the best groups in every generation, default 0.1
- n_jobs: the number for running procedure in parallel, default 4
- verbose: the flag for showing the verbose messages, default False


### Sampling

```python
# Example

from genetics import GA

# the sample_result is a 2D numpy matrix, which is the result after sampling
# the sample_genes is the gene used for selecting instances, just ignore it if you don't need it
# the sample_scores is the final score when doing validation in valid set
(sample_result, sample_genes, sample_scores) = GA(train, valid, RandomForestClassfier).select_instance()

# Or you can do sampling by calling this
(sample_result, sample_genes, sample_scores) = GA(train, valid, RandomForestClassfier).select(axis=0)

```

### Feature Selection

```python
# Example

from genetics import GA

# the sample_result is a 2D numpy matrix, which is the result after selecting feature
# the sample_genes is the gene used for selecting features, just ignore it if you don't need it
# the sample_scores is the final score when doing validation in valid set
(sample_result, sample_genes, sample_scores) = GA(train, valid, RandomForestClassfier).select_feature()

# Or you can select features by calling this
(sample_result, sample_genes, sample_scores) = GA(train, valid, RandomForestClassfier).select(axis=1)

```

# Citation

```
Zeng X, Yuan S, Huang X, et al. 
Identification of cytokine via an improved genetic algorithm[J]. 
Frontiers of Computer Science, 2015, 9(4): 643-651.
```
