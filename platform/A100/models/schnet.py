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


def validate(
    batch_size=32,
    hidden_channels=128,
    num_filters=128,
    num_interactions=6,
    max_num_neighbors=32,
    record=None,
    mode=None,
    ops_save_dir=None,
    profiler_dir=None,
    datatype=None,
    device=None
):

    path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "QM9")
    dataset = QM9(path)

    model = SchNet(
        hidden_channels=hidden_channels,
        num_filters=num_filters,
        num_interactions=num_interactions,
        max_num_neighbors=max_num_neighbors,
    )

    params = sum(p.numel() for p in model.parameters())
    print(f"Num parameters: {params}")

    model = model.to(device)
    if datatype == "fp32":
        model = model.to(torch.float32)
    elif datatype == "fp16":
        model = model.to(torch.float16)
    else:
        print(f"Data type {datatype} is invalid")

    my_schedule = schedule(wait=5, warmup=1, active=4)  # repeat=1
    model = model.eval()

    running_loss = 0
    valtimes = []
    valtime_count = 0

    with torch.no_grad():

        with profile(
            activities=[record],
            record_shapes=False,
            schedule=my_schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir),
        ) as prof:
            with record_function("model_inference"):

                for target in range(12):

                    # TODO: cast dataset to datatype
                    if datatype != "fp32":
                        raise ValueError("Sorry, only fp32 supported for now")

                    loader = DataLoader(dataset, batch_size=batch_size)

                    maes = []
                    for step, data in enumerate(loader):

                        if step >= (5 + 1 + 4) * 1:
                            break

                        data = data.to(device)

                        starttime = timeit.default_timer()

                        pred = model(data.z, data.pos, data.batch)
                        mae = (pred.view(-1) - data.y[:, target]).abs()

                        endtime = timeit.default_timer()
                        valtimes.append(endtime - starttime)
                        valtime_count += 1

                        prof.step()

                        maes.append(mae)

                    mae = torch.cat(maes, dim=0)
                    running_loss += (mae)

                param = None
                if mode == "batch_size":
                    param = batch_size
                elif mode == "hidden_channels":
                    param = hidden_channels
                elif mode == "num_filters":
                    param = num_filters
                elif mode == "num_interactions":
                    param = num_interactions
                elif mode == "max_num_neighbors":
                    param = max_num_neighbors
                else:
                    raise ValueError(f"Warning, invalid mode: {mode}")

                prof_type = "cpu" if record == ProfilerActivity.CPU else "cuda"
                df = prof_to_df(prof)
                df.to_csv(
                    f"{ops_save_dir}/schnet/schnet-{prof_type}-{mode}-{param}.csv",
                    index=False,
                )

    valtime = sum(valtimes)/valtime_count

    return valtime, params


@click.command()
@click.option('--datatype', default="fp32", help='Percision used for data and model weights. One of "fp32" or "fp16".')
@click.option('--mode', default=None, help='Mode used for benchmarking. One of "batch_size", "hidden_channels", "num_filters", "num_interactions", "max_num_neighbors", or "all".')
@click.option('--ops_save_dir', default="/lus/grand/projects/datascience/gnn-dataflow/profiling_data/", help='Location to save ops profiles. Path will be appended with "--datatype"')
@click.option('--latency_save_dir', default="./logs", help='Location to save ops profiles. Path will be appended with "--datatype"')
@click.option('--profiler_dir', default="./runs/profiler", help="Path for profiler results. Can be loaded into tensorboard visualization.")
@click.option('--device', default="cuda", help='Device used for profiling. One of "cuda" or "cpu".')
@click.option('--seed', default=0, help='Random seed used. Default is 0.')
def cli(datatype, mode, ops_save_dir, latency_save_dir, profiler_dir, device, seed):

    # Macros
    DATATYPE = datatype
    MODE = mode  # batch_size, width, depth
    OPS_SAVE_DIR = (
        f"{ops_save_dir}/{DATATYPE}"
    )
    LATENCY_SAVE_DIR = f"{latency_save_dir}/{DATATYPE}"
    PROFILER_DIR = profiler_dir
    DEVICE = device
    RECORD = ProfilerActivity.CUDA if device.lower() == "cuda" else "cpu"

    torch.manual_seed(seed)

    valtimes = []
    params_lst = []
    batch_size_lst = []

    """
    Batch size
    """
    valtimes = []
    params_lst = []
    batch_size_lst = []

    if MODE in ["all", "batch_size"]:
        batchsizes = [2 ** x for x in range(0, 14)]
        for idx, batchsize in enumerate(batchsizes):

            print(f"Iteration: {idx+1}, batchsize: {batchsize}")

            valtime, params = validate(
                batch_size=batchsize,
                record=RECORD,
                mode=MODE,
                ops_save_dir=OPS_SAVE_DIR,
                profiler_dir=PROFILER_DIR,
                datatype=DATATYPE,
                device=DEVICE
            )

            valtimes.append(valtime)
            params_lst.append(params)
            batch_size_lst.append(batchsize)

        results = {"val_times": valtimes, "params": params_lst, "batch_size": batch_size_lst}

        df = pd.DataFrame(results)
        df.to_csv(f"{LATENCY_SAVE_DIR}/A100/schnet/{MODE}.csv")

    """
    Hidden channels
    """
    valtimes = []
    params_lst = []
    batch_size_lst = []

    if MODE in ["all", "hidden_channels"]:
        possible_param_ws = [2 ** x for x in range(1, 16)]

        for idx, param_w in enumerate(possible_param_ws):

            print(f"Iteration: {idx+1}, param_w: {param_w}")

            valtime, params = validate(
                hidden_channels=param_w,
                record=RECORD,
                mode=MODE,
                ops_save_dir=OPS_SAVE_DIR,
                profiler_dir=PROFILER_DIR,
                datatype=DATATYPE,
                device=DEVICE
            )
            valtimes.append(valtime)
            params_lst.append(params)
            batch_size_lst.append(32)

        results = {"val_times": valtimes, "params": params_lst, "batch_size": batch_size_lst}

        df = pd.DataFrame(results)
        df.to_csv(f"{LATENCY_SAVE_DIR}/A100/schnet/{MODE}.csv")

    """
    Num filters
    """
    valtimes = []
    params_lst = []
    batch_size_lst = []

    if MODE in ["all", "num_filters"]:
        possible_param_nfilters = [2 ** x for x in range(0, 16)]

        for idx, param_f in enumerate(possible_param_nfilters):

            print(f"Iteration: {idx+1} param_f: {param_f}")

            valtime, params = validate(
                num_filters=param_f,
                record=RECORD,
                mode=MODE,
                ops_save_dir=OPS_SAVE_DIR,
                profiler_dir=PROFILER_DIR,
                datatype=DATATYPE,
                device=DEVICE
            )
            valtimes.append(valtime)
            params_lst.append(params)
            batch_size_lst.append(32)

        results = {"val_times": valtimes, "params": params_lst, "batch_size": batch_size_lst}

        df = pd.DataFrame(results)
        df.to_csv(f"{LATENCY_SAVE_DIR}/A100/schnet/{MODE}.csv")

    """
    Num interactions
    """
    valtimes = []
    params_lst = []
    batch_size_lst = []

    if MODE in ["all", "num_interactions"]:
        # LINEAR
        possible_param_ninteractions = [10 * x for x in range(1, 40)]  # technically have not hit upper bound here

        for idx, param_i in enumerate(possible_param_ninteractions):

            print(f"Iteration: {idx+1} param_i: {param_i}")

            valtime, params = validate(
                num_interactions=param_i,
                record=RECORD,
                mode=MODE,
                ops_save_dir=OPS_SAVE_DIR,
                profiler_dir=PROFILER_DIR,
                datatype=DATATYPE,
                device=DEVICE
            )
            valtimes.append(valtime)
            params_lst.append(params)
            batch_size_lst.append(32)

        results = {"val_times": valtimes, "params": params_lst, "batch_size": batch_size_lst}

        df = pd.DataFrame(results)
        df.to_csv(f"{LATENCY_SAVE_DIR}/A100/schnet/{MODE}.csv")

    """
    Max num neighbors
    """
    valtimes = []
    params_lst = []
    batch_size_lst = []

    if MODE in ["all", "max_num_neighbors"]:
        possible_param_mnn = [2 ** x for x in range(0, 23)]

        for idx, param_mnn in enumerate(possible_param_mnn):

            print(f"Iteration: {idx+1} param_mnn: {param_mnn}")

            valtime, params = validate(
                max_num_neighbors=param_mnn,
                record=RECORD,
                mode=MODE,
                ops_save_dir=OPS_SAVE_DIR,
                profiler_dir=PROFILER_DIR,
                datatype=DATATYPE,
                device=DEVICE
            )
            valtimes.append(valtime)
            params_lst.append(params)
            batch_size_lst.append(32)

        results = {"val_times": valtimes, "params": params_lst, "batch_size": batch_size_lst}

        df = pd.DataFrame(results)
        df.to_csv(f"{LATENCY_SAVE_DIR}/A100/schnet/{MODE}.csv")


if __name__ == "__main__":
    cli()
