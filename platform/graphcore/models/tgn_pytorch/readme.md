# Running TGN on Graphcore POD System at ANL

Temporal graph networks for link prediction in dynamic graphs, based on [`examples/tgn.py`](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/tgn.py) from PyTorch-Geometric, optimised for Graphcore's IPU.

Refer to the [original readme file](./README_original.md) that came with repository. 


## Setup

* Poplar and Popart Setup
```bash
  source /software/graphcore/poplar_sdk/3.1.0/poplar-ubuntu_20_04-3.1.0+6824-9c103dc348/enable.sh
  source /software/graphcore/poplar_sdk/3.1.0/popart-ubuntu_20_04-3.1.0+6824-9c103dc348/enable.sh
```
Either run these two commands each time or add them to your `bash_profile`

* Initial Environment and dependancy Setup: 
  *  TGN example uses PyTorch. The dependacnies are different than other GNN models. To keep everythig clean, create a new virtual environment. 
    ```bash
      virtualenv -p python3 ~/workspace/poptorch_env
      source ~/workspace/poptorch_env/bin/activate
      pip3 install -U pip

      pip3 install /software/graphcore/poplar_sdk/3.1.0/poptorch-3.1.0+98660_0a383de63f_ubuntu_20_04-cp38-cp38-linux_x86_64.whl 
    ```
  *. Install the Python requirements:
    ```bash
    pip3 install -r requirements.txt
    ```

* Going forward just activate the environment. 
```bash
  source ~/workspace/poptorch_env/bin/activate
```

## Running 


* Running 
```bash
python3 -m examples_utils benchmark --spec benchmarks.yml
```

* Output of this is in [log_benchmark_run](./log_benchmark_run/)

Or to run a specific benchmark in the `benchmarks.yml` file provided:


## Profiling 

* PopVision System Analyzer

  * Enabling instrumentation 
    `PVTI_OPTIONS='{"enable":"true", "directory": "system_profile"}' python3 linear_net.py`
  
  * The output log files are in [log_system_profile](./log_system_profile)

  * The profiling output files generated for this are in shared directory `/lambda_stor/homes/sraskar/gnn-dataflow-share/tgn/system_profile`on Graphcore POD system.
  
  * Open the `.pvti` file using PopVision System Analyzer on your local system.
    * Create a ssh tunnnel `ssh gc-pod -L 8090:127.0.0.1:22`. 

* PopVision Graph Analyzer

  * Enabling instrumentation 
  `POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"./graph_profile"}' python train.py`

  * The output log files are in [log_graph_profile](./log_graph_profile)

  * The profiling output files generated for this are in shared directory `/lambda_stor/homes/sraskar/gnn-dataflow-share/tgn/graph_profile`on Graphcore POD system.
  
  * Open the `.pop` file using PopVision Graph Analyzer on your local system. 
    * Create a ssh tunnnel `ssh gc-pod -L 8090:127.0.0.1:22`. 