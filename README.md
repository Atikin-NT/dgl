# Repository with an example of training a GCN using DGL


## Introduction


This repository provides an example of training a graph neural network 
using the DGL framework on the [Cora][cora] dataset.


The Cora dataset consists of 2708 scientific publications classified into 
one of seven classes. The citation network consists of 5429 links. 
Each publication in the dataset is described by a 0/1-valued word vector 
indicating the absence/presence of the corresponding word from 
the dictionary.


**Goal**: predict which class an article belongs to.

## Repo structure

- `dgl_cora.ipynb` file with a description of the model architecture, 
  its training, as well as an inference.

- `models` directory for storing a model file in **pt** format , 
  as well as a python file with architecture.

  - `GCN.py` file with a description of the model architecture.
  - `model.pt` model file in **pt** format.

## Usage

This repository is used to provide files to the [dl-benchmark][dl-benchmark] repository.

<!-- LINKS -->
[dl-benchmark]: https://github.com/itlab-vision/dl-benchmark
[cora]: https://relational.fit.cvut.cz/dataset/CORA