# Running SchNet on Graphcore POD System at ANL

## Setup

* Poplar and Popart Setup
```bash
  source /software/graphcore/poplar_sdk/3.1.0/poplar-ubuntu_20_04-3.1.0+6824-9c103dc348/enable.sh
  source /software/graphcore/poplar_sdk/3.1.0/popart-ubuntu_20_04-3.1.0+6824-9c103dc348/enable.sh
```
Either run these two commands each time or add them to your `bash_profile`

* Initial Environment and dependancy Setup: 
  *  Schenet example uses PyTorch. The dependacnies are different than other GNN models. To keep everythig clean, create a new virtual environment. 
    ```bash
      virtualenv -p python3 ~/workspace/schnet_env
      source ~/workspace/schnet_env/bin/activate
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
python3 train.py
```

* Output of this is in [output.log](./output.log) file.


## Profiling 

* PopVision System Analyzer

  * Enabling instrumentation 
    `PVTI_OPTIONS='{"enable":"true", "directory": "system_profile"}' python3 linear_net.py`

  * The output of this is in shared directory `/lambda_stor/homes/sraskar/gnn-dataflow-share/schnet/system_profile`on Graphcore POD system.
  
  * Open the `.pvti` file using PopVision System Analyzer on your local system.

* PopVision Graph Analyzer

  * Enabling instrumentation 
  `POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"./graph_profile"}' python train.py`

  * * The output of this is in shared directory `/lambda_stor/homes/sraskar/gnn-dataflow-share/schnet/graph_profile`on Graphcore POD system.
  
  * * Open the `.pop` file using PopVision Graph Analyzer on your local system. 