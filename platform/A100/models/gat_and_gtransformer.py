import torch
from torch_geometric.nn import (
    GINConv,
    FiLMConv,
    GATv2Conv,
    TransformerConv,
    global_mean_pool,
)

from torch_geometric.loader import DataLoader

from ogb.utils import smiles2graph
from ogb.lsc import PygPCQM4Mv2Dataset, PCQM4Mv2Evaluator

import timeit
import tqdm
import pandas as pd

import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity, schedule

from util import prof_to_df
import os
from graph_transformer_pytorch import GraphTransformer

import click


class GNNREG(torch.nn.Module):
    def __init__(
        self, model_name, input_dim, hidden_dim, dropout, num_conv_layers, heads
    ):

        super(GNNREG, self).__init__()

        self.model_name = model_name

        self.dropout = dropout
        self.num_layers = num_conv_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim, heads))
        self.lns = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim, heads))
            self.lns.append(torch.nn.LayerNorm(hidden_dim))

        self.conv_dropout = torch.nn.Dropout(p=self.dropout)
        self.ReLU = torch.nn.ReLU()

        # post-message-passing
        self.post_mp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 1),
        )

    def build_conv_model(self, input_dim, hidden_dim, heads):
        if self.model_name == "gTransformer":
            return TransformerConv(
                in_channels=input_dim,
                out_channels=hidden_dim,
                heads=heads,
                concat=False,
            )
        elif self.model_name == "GATv2":
            return GATv2Conv(
                in_channels=input_dim,
                out_channels=hidden_dim,
                heads=heads,
                concat=False,
            )
        else:
            raise Exception(f"{self.model_name} not a valid model")

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.to(torch.float32)
        edge_index = edge_index.to(torch.long)
        if data.num_node_features == 0:
            print("Warning: No node features detected.")
            x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.ReLU(x)
            x = self.conv_dropout(x)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        x = global_mean_pool(x, batch)
        x = self.post_mp(x)

        return x


def evaluate_epoch(
    val_loader,
    model,
    bs,
    width,
    depth,
    record,
    mode,
    ops_save_dir,
    model_name,
):

    my_schedule = schedule(wait=5, warmup=1, active=4)  # repeat=1

    model = model.eval()

    running_loss = 0
    valtimes = []
    valtime_count = 0

    # cntr = 0
    with torch.no_grad():

        with profile(
            activities=[record],
            record_shapes=False,
            schedule=my_schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./runs/profiler"),
        ) as prof:
            with record_function("model_inference"):

                for step, datum in enumerate(val_loader):

                    if step >= (5 + 1 + 4) * 1:
                        break
        
                    starttime = timeit.default_timer()

                    logits = model(datum)

                    prediction = torch.squeeze(logits)
                    my_loss = torch.nn.functional.mse_loss(prediction, datum.y)

                    running_loss += my_loss

                    endtime = timeit.default_timer()
                    valtimes.append(endtime - starttime)
                    valtime_count += 1

                    prof.step()

                param = None
                if mode == "batch_size":
                    param = bs
                elif mode == "width":
                    param = width
                elif mode == "depth":
                    param = depth
                else:
                    raise ValueError(f"Warning, invalid mode: {mode}")

                prof_type = "cpu" if record == ProfilerActivity.CPU else "cuda"
                df = prof_to_df(prof)
                df.to_csv(
                    f"{ops_save_dir}/{model_name}/{model_name}-{prof_type}-{mode}-{param}.csv",
                    index=False,
                )

        valtime = sum(valtimes)/valtime_count

        return running_loss, valtime


def main(
    node_feature_size=9,
    hidden_dim=64,
    num_conv_layers=4,
    num_heads=32,
    batch_size=32,
    record=ProfilerActivity.CUDA,
    mode=None,
    ops_save_dir=None,
    model_name=None,
    datatype=None,
    device=None
):

    model = GNNREG(
        model_name=model_name,
        input_dim=node_feature_size,
        hidden_dim=hidden_dim,
        dropout=0,
        num_conv_layers=num_conv_layers,
        heads=num_heads,
    )

    if datatype == "fp32":
        model = model.to(torch.float32)

    elif datatype == "fp16":
        model = model.to(torch.float16)
    else:
        print(f"Data type {datatype} is invalid")

    model = model.to("cuda")

    params = sum(p.numel() for p in model.parameters())
    print(f"Num parameters: {params}")

    # get dataset, splits
    dataset = PygPCQM4Mv2Dataset(root="./data", smiles2graph=smiles2graph)

    split_dict = dataset.get_idx_split()
    valid_idx = split_dict["valid"]

    valid_dataset = dataset[valid_idx]
    valid_dataset.data.edge_index = valid_dataset.data.edge_index.to("cuda")
    if datatype == "fp32":
        valid_dataset.data.edge_attr = valid_dataset.data.edge_attr.to(
            device=device, dtype=torch.float32
        )
        valid_dataset.data.x = valid_dataset.data.x.to(
            device=device, dtype=torch.float32
        )
        valid_dataset.data.y = valid_dataset.data.y.to(
            device=device, dtype=torch.float32
        )
    if datatype == "fp16":
        valid_dataset.data.edge_attr = valid_dataset.data.edge_attr.to(
            device=device, dtype=torch.float16
        )
        valid_dataset.data.x = valid_dataset.data.x.to(
            device=device, dtype=torch.float16
        )
        valid_dataset.data.y = valid_dataset.data.y.to(
            device=device, dtype=torch.float16
        )

    # valid_dataset = valid_dataset.data.to("cuda")
    # torch.cuda.synchronize()

    va_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        # num_workers=2,
        # prefetch_factor=batch_size,
    )

    running_loss, valtime = evaluate_epoch(
        va_loader,
        model,
        bs=batch_size,
        width=hidden_dim,
        depth=num_conv_layers,
        record=record,
        mode=mode,
        ops_save_dir=ops_save_dir,
        model_name=model_name,
    ) # type: ignore

    return valtime, params


# command line
@click.command()
@click.option('--datatype', default="fp32", help='Percision used for data and model weights. One of "fp32" or "fp16".')
@click.option('--arch', default=None, help='Architecture used for benchmarking. One of "gTransformer" or "GATv2".')
@click.option('--mode', default=None, help='Mode used for benchmarking. One of "batch_size", "width", or "depth".')
@click.option('--ops_save_dir', default="/lus/grand/projects/datascience/gnn-dataflow/profiling_data/", help='Location to save ops profiles. Path will be appended with "--datatype"')
@click.option('--latency_save_dir', default="./logs", help='Location to save ops profiles. Path will be appended with "--datatype"')
@click.option('--device', default="cuda", help='Device used for profiling. One of "cuda" or "cpu".')
@click.option('--seed', default=0, help='Random seed used. Default is 0.')
def cli(datatype, arch, mode, ops_save_dir, latency_save_dir, device, seed):
    # Macros
    DATATYPE = datatype
    ARCH = arch
    MODE = mode  # batch_size, width, depth
    OPS_SAVE_DIR = (
        f"{ops_save_dir}/{DATATYPE}"
    )
    LATENCY_SAVE_DIR = f"{latency_save_dir}/{DATATYPE}"
    DEVICE = device
    RECORD = ProfilerActivity.CUDA if device.lower() == "cuda" else "cpu"

    torch.manual_seed(seed)

    valtimes = []
    params_lst = []

    """
    Batch size
    """
    if MODE == "batch_size":
        batchsizes = [2 ** x for x in range(0, 11)]
        for batchsize in batchsizes:  # 10, 12000

            print(f"Iteration {x+1}")

            valtime, params = main(
                batch_size=batchsize,
                record=RECORD,
                mode=MODE,
                ops_save_dir=OPS_SAVE_DIR,
                model_name=ARCH,
                datatype=DATATYPE,
                device=DEVICE
            )

            valtimes.append(valtime)
            params_lst.append(params)

        results = {"val_times": valtimes, "params": params_lst}

        df = pd.DataFrame(results)
        df.to_csv(f"{LATENCY_SAVE_DIR}/A100/{ARCH}/{MODE}.csv")

    """
    Parameter width
    """
    if MODE == "width":
        possible_param_ws = [2 ** x for x in range(12)]

        for param_w in possible_param_ws:

            print(f"Iteration {x+1}")

            valtime, params = main(
                hidden_dim=param_w,
                record=RECORD,
                mode=MODE,
                ops_save_dir=OPS_SAVE_DIR,
                model_name=ARCH,
                datatype=DATATYPE,
                device=DEVICE
            )
            valtimes.append(valtime)
            params_lst.append(params)

        results = {"val_times": valtimes, "params": params_lst}

        df = pd.DataFrame(results)
        df.to_csv(f"{LATENCY_SAVE_DIR}/A100/{ARCH}/{MODE}.csv")

    """
    Parameter length
    """
    if MODE == "depth":
        possible_param_ls = [2 ** x for x in range(10)]

        for param_l in possible_param_ls:

            print(f"Iteration {x+1}")

            valtime, params = main(
                num_conv_layers=param_l,
                record=RECORD,
                mode=MODE,
                ops_save_dir=OPS_SAVE_DIR,
                model_name=ARCH,
                datatype=DATATYPE,
                device=DEVICE
            )
            valtimes.append(valtime)
            params_lst.append(params)

        results = {"val_times": valtimes, "params": params_lst}

        df = pd.DataFrame(results)
        df.to_csv(f"{LATENCY_SAVE_DIR}/A100/{ARCH}/{MODE}.csv")




if __name__ == "__main__":
    cli()
