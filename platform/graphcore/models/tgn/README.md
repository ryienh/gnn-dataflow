# TGN: Temporal Graph Networks

An implementation of [TGN](https://arxiv.org/abs/2006.10637) in TensorFlow 1 for IPU. <br><br>
Run our Temporal Graph Networks on JODIE Wikipedia dataset on Paperspace.
<br>
[![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/gradient-ai/Graphcore-Tensorflow1?machine=Free-IPU-POD16&container=graphcore%2Ftensorflow-jupyter%3A1-amd-2.6.0-ubuntu-18.04-20220804&file=%2Fget-started%2FTrainingTGN.ipynb)

## About

The temporal graph network is a dynamic GNN that can predict link edit operations in dynamic graphs. The model includes a sparsely accessed memory and a [Graph Transformer](https://arxiv.org/abs/2009.03509) layer. Following the authors, we apply it to a dataset of Wikipedia edits from [JODIE](http://snap.stanford.edu/jodie). The task is to predict edit events that connect users with pages. We apply it in the transductive setting where the same set of users and pages is used for validation and testing (although the edit events are distinct).

The JODIE-Wikipedia dataset is available from [snap.stanford.edu](http://snap.stanford.edu/jodie/#datasets), but note that it will be automatically downloaded when `run_tgn.py` is first run.

We reproduce the behaviour of PyTorch Geometric [`examples/tgn.py`](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/tgn.py). While the main body of our implementation is distinct, we share code for data loading and nearest neighbour preprocessing. We have also made a few modifications:

 - Use lower precision (`tf.float16`) where possible.
 - Recompute the memory at validation/test time in the same way as training (note: this should not change the results).
 - Concatenate the memory payload to reduce the number of `tf.gather` calls.
 - When benchmarking, cache the dataset (reusing negative samples) and only validate at the end of training.


## Quick start

**1. Download and install the Poplar SDK following the Getting Started guide for your IPU system.**

**2. Create a virtual environment and install the appropriate Graphcore TensorFlow 1.15 wheel from inside the SDK directory:**

```shell
virtualenv --python python3 .venv
source .venv/bin/activate
source <path to enable.sh for Poplar from the Poplar SDK>
pip install <path to the tensorflow-1 wheel from the Poplar SDK>
pip install -r requirements.txt
```

**3. Run training, profiling or benchmarking:**

```shell
# Training (should reach 0.97 auc, 0.97 ap, 0.35 training loss)
# NOTE: On first run, this will download the JODIE-Wikipedia dataset (approx 500M) - this may take a while
python run_tgn.py

# Profiling
env POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"report"}' python run_tgn.py -m profile
```

## Running and benchmarking

To run a tested and optimised configuration and to reproduce the performance shown on our [performance results page](https://www.graphcore.ai/performance-results), please follow the setup instructions in this README to setup the environment, and then use the `examples_utils` module (installed automatically as part of the environment setup) to run one or more benchmarks. For example:

```python
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file>
```

Or to run a specific benchmark in the `benchmarks.yml` file provided:

```python
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file> --benchmark <name of benchmark>
```

For more information on using the examples-utils benchmarking module, please refer to [the README](https://github.com/graphcore/examples-utils/blob/master/examples_utils/benchmarks/README.md).

## File structure

| Path | Description |
| ---- | ----------- |
| [`run_tgn.py`](run_tgn.py) | training loop and entry point for running the TGN |
| [`model.py`](model.py) | TensorFlow model and single training step |
| [`optimiser.py`](optimiser.py) | low-precision Adam optimiser |
| [`utils.py`](utils.py) | helpers for defining portable training loops |
| [`dataloader.py`](dataloader.py) | dataset download, batching and neighbour sampling |
| [`requirements.txt`](requirements.txt), [`setup.cfg`](setup.cfg) | setup & development |
| [`tests/`](tests) | unit tests |
| `data/JODIE/` | dataset, automatically downloaded when running `run_tgn.py` |


## References

 - Temporal Graph Networks for Deep Learning on Dynamic Graphs, Rossi, E., et al., https://arxiv.org/abs/2006.10637
 - Fast Graph Representation Learning with PyTorch Geometric, Fey, M. and Lenssen, J.E., https://arxiv.org/abs/1903.02428
 - Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks, Kumar, S., Zhang, X. and Leskovec, J., https://arxiv.org/abs/1908.01207
 - Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification, Shi, Y., et al., https://arxiv.org/abs/2009.03509


## License

This example is licensed under the MIT license - see the LICENSE file at the top-level of this repository.

This directory includes derived work from the following:

> PyTorch Geometric, https://github.com/pyg-team/pytorch_geometric/
>
> Copyright (c) 2021 Matthias Fey, Jiaxuan You <matthias.fey@tu-dortmund.de, jiaxuan@cs.stanford.edu>
>
> Licensed under the MIT License
