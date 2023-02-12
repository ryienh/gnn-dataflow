import os
import os.path as osp

import torch
import poptorch
import pandas as pd
import py3Dmol

from periodictable import elements
from torch_geometric.datasets import QM9
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import to_fixed_size
from torch_geometric.nn.models import SchNet
from tqdm import tqdm

from torch_geometric.nn import MessagePassing

from pyg_schnet_util import (TrainingModule, KNNInteractionGraph, prepare_data,
                             padding_graph, create_dataloader, optimize_popart)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

poptorch.setLogLevel('ERR')
executable_cache_dir = os.getenv("POPLAR_EXECUTABLE_CACHE_DIR", "/tmp/exe_cache/") + "/pyg-schnet"
dataset_directory = os.getenv("DATASET_DIR", 'data')
num_ipus = os.getenv("NUM_AVAILABLE_IPU", "16")
num_ipus = min(int(num_ipus), 16) # QM9 is too small to benefit from additional scaling

qm9_root = osp.join(dataset_directory, 'qm9')
dataset = QM9(qm9_root)

print(len(dataset))

cutoff = 10.0

num_test = 10000
num_val = 10000
torch.manual_seed(0)
dataset.transform = prepare_data
dataset = dataset.shuffle()
test_dataset = dataset[:num_test]
val_dataset = dataset[num_test:num_test + num_val]
train_dataset = dataset[num_test + num_val:]

print(f"Number of test molecules: {len(test_dataset)}\n"
      f"Number of validation molecules: {len(val_dataset)}\n"
      f"Number of training molecules: {len(train_dataset)}")

batch_size = 8
replication_factor = num_ipus
device_iterations = 32
gradient_accumulation = 16 // num_ipus
learning_rate = 1e-4
num_epochs = 30

options = poptorch.Options()
options.enableExecutableCaching(executable_cache_dir)
options.outputMode(poptorch.OutputMode.All)
options.deviceIterations(device_iterations)
options.replicationFactor(replication_factor)
options.Training.gradientAccumulation(gradient_accumulation);

additional_optimizations = True

if additional_optimizations:
    options = optimize_popart(options)

train_loader = create_dataloader(train_dataset,
                                 options,
                                 batch_size,
                                 shuffle=True)
torch.manual_seed(0)
knn_graph = KNNInteractionGraph(cutoff=cutoff, k=28)
model = SchNet(cutoff=cutoff, interaction_graph=knn_graph)
model.train()
model = TrainingModule(model,
                       batch_size=batch_size,
                       replace_softplus=additional_optimizations)
optimizer = poptorch.optim.AdamW(model.parameters(), lr=learning_rate)
training_model = poptorch.trainingModel(model, options, optimizer)

data = next(iter(train_loader))
training_model.compile(*data)

train = []

for epoch in range(num_epochs):
    bar = tqdm(train_loader)
    for i, data in enumerate(bar):
        _, mini_batch_loss = training_model(*data)
        loss = float(mini_batch_loss.mean())
        train.append({'epoch': epoch, 'step': i, 'loss': loss})
        bar.set_description(f"Epoch {epoch} loss: {loss:0.6f}")

training_model.detachFromDevice()