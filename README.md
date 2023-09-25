# gnn-dataflow

## Running end-to-end profiles on A100

These steps detail running end-to-end profiles the thetagpu platform:

1. Request compute:
```
qsub -I -A [PROJ_NAME] -n 1 -t 60 -q single-gpu
```

2. Navigate to directory, activate conda env, export proper runtime libs:
```
cd /lus/PATH/TO/REPO
module load conda
conda activate /lus/PATH/TO/CONDA_ENV
export PATH=/soft/thetagpu/cuda/cuda-11.8.0/bin:$PATH
export LD_LIBRARY_PATH=/soft/thetagpu/cuda/cuda-11.8.0/bin:$PATH
```
3. Run profile scripts

```
python platform/A100/models/gat_and_gtransformer.py
```
(and similar for tgn, schnet, GPS++)

4. Results can be logged to custom path but will appear in `./logs` and `./runs/profiler` by default. 
