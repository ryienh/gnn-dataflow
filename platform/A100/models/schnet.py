import click
import os.path as osp

import torch
from tqdm import tqdm

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet


import timeit
import pandas as pd

import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity, schedule

from util import prof_to_df

@click.command()
@click.option('--batch_size', default=32, help='Batch size used for forward inference')
@click.option('--hidden_channels', default=128, help='Hidden embedding size')
@click.option('--num_filters', default=128, help='The number of filters used')
@click.option('--num_interactions', default=6, help='The number of interaction blocks')
@click.option('--max_num_neighbors', default=32, help='The maximum number of neighbors to collect for each node within a set cutoff distance.')
def main(
    batch_size=32,
    hidden_channels=128,
    num_filters=128,
    num_interactions=6,
    max_num_neighbors=32,
):

    path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "QM9")
    dataset = QM9(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for target in range(12):
        _, datasets = SchNet.from_qm9_pretrained(path, dataset, target)  # model
        train_dataset, val_dataset, test_dataset = datasets

        model = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            max_num_neighbors=max_num_neighbors,
        )

        model = model.to(device)
        loader = DataLoader(test_dataset, batch_size=batch_size)

        maes = []
        for data in tqdm(loader):
            data = data.to(device)
            with torch.no_grad():
                pred = model(data.z, data.pos, data.batch)
            mae = (pred.view(-1) - data.y[:, target]).abs()
            maes.append(mae)

        mae = torch.cat(maes, dim=0)

        # Report meV instead of eV.
        mae = 1000 * mae if target in [2, 3, 4, 6, 7, 8, 9, 10] else mae

        print(f"Target: {target:02d}, MAE: {mae.mean():.5f} ± {mae.std():.5f}")


if __name__ == "__main__":

    # Macros
    DATATYPE = "fp32"
    MODE = "batch_size"  # batch_size, hidden_channels, num_filters, num_interactions, max_num_neighbors
    OPS_SAVE_DIR = (
        f"/lus/grand/projects/datascience/gnn-dataflow/profiling_data/{DATATYPE}"
    )
    LATENCY_SAVE_DIR = f"./logs/{DATATYPE}"
    RECORD = ProfilerActivity.CUDA  # ProfilerActivity.CUDA, ProfilerActivity.CPU

    torch.manual_seed(0)

    valtimes = []
    params_lst = []

    """
    Batch size
    """
    if MODE == "batch_size":
        batchsizes = [2 ** x for x in range(0, 9)]
        for batchsize in batchsizes:  # 10, 12000

            valtimes, params = main(batch_size=batchsize)

            # traintimes.append(traintime)
            valtimes.append(valtime)
            params_lst.append(params)

        results = {"val_times": valtimes, "params": params_lst}

        df = pd.DataFrame(results)
        df.to_csv(f"{LATENCY_SAVE_DIR}/A100/{ARCH}/{MODE}.csv")

    """
    Hidden channels
    """
    if MODE == "hidden_channels":
        hidden_channels = [2 ** x for x in range(0, 9)]
        for hidden_channel in hidden_channels:  # 10, 12000

            valtime, params = main(hidden_channels=hidden_channel)

            # traintimes.append(traintime)
            valtimes.append(valtime)
            params_lst.append(params)

        results = {"val_times": valtimes, "params": params_lst}

        df = pd.DataFrame(results)
        df.to_csv(f"{LATENCY_SAVE_DIR}/A100/{ARCH}/{MODE}.csv")
