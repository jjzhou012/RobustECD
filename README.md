# RobustECD: Robust Enhancement of Community Detection

This is a Python implementation of RobustECD, as described in the following:

> RobustECD: Robust Enhancement of Community Detection in Complex Networks

## Requirements

The code is tested on Ubuntu 16.04 and Windows 10 with the following components:

### Software

- Python 3.7
- NetworkX 2.4
- SciPy 1.4.1 
- NumPy 1.18.1
- python-igraph 0.7.1.post6

### Datasets

Real-world benchmark networks:

- `Karate`, `Polbooks`, `Football`, `Polblogs`

Large-scale real-world networks from [Stanford Large Network Dataset Collection](http://snap.stanford.edu/data/):

- `Amazon`, `DBLP`

Adversarial networks generated via adversarial attack on benchmark networks:

- `Karate_noise`, `Polbooks_noise`, `Football_noise`, `Polblogs_noise`

## Usage

- *RobustECD-SE*: execute the following `bash` commands in the same directory where the code resides:

  ```bash
  $ python exp_revsel.py --bmname karate --cdm LOU --randomSample 1 --sampleRatio 1.6
  ```

- *RobustECD-GA*: execute the following `bash` commands in the same directory where the code resides:

  ```bash
  $ python exp_rega.py --bmname karate --cdm INF --iter 500 -aR 0.16 -dR 0.16
  ```

Common Parameters:

- `bmname`: name of dataset
  - benchmark: `karate`, `polbooks`, `football`, `polblogs`
  - large-scale subgraph: `amazon-sub`, `dblp-sub`
  - adversarial networks: `karate_noise`, `polbooks_noise`, `football_noise`, `polblogs_noise`
- `cdm`: community detection method
  - Infomap: `INF`
  - Fast Greedy: `FG`
  - WalkTrap: `WT`
  - Louvain: `LOU`
  - Label Propagation: `LP`
  - Node2vec+Kmeans: `n2v_km`



## Citation

If you find this work useful, please cite the following:

```
@
```

