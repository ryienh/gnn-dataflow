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


class GATREG(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, num_conv_layers, heads):

        super(GATREG, self).__init__()

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
        return TransformerConv(
            in_channels=input_dim, out_channels=hidden_dim, heads=heads, concat=False
        )

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
    device,
    bs,
    width,
    depth,
    record,
    mode,
    ops_save_dir,
    model_name,
    nodes,
    edges,
    mask,
):

    my_schedule = schedule(wait=5, warmup=1, active=4)  # repeat=1

    model = model.eval()

    running_loss = 0
    predictions = []
    labels = []

    # cntr = 0
    with torch.no_grad():

        with profile(
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            record_shapes=False,
            schedule=my_schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./runs/profiler"),
        ) as prof:
            with record_function("model_inference"):

                for step in range(100):

                    if step >= (5 + 1 + 4) * 1:
                        break

                    # X.y = X.y.to(torch.float32)
                    # nodes = X.x
                    # edges = X.edge_attr

                    logits, edges = model(nodes, edges, mask)

                    prediction = torch.squeeze(logits)
                    # my_loss = torch.nn.functional.mse_loss(prediction, X.y)

                    # prof.step()

                    # cntr += 1
                    # if cntr == 10:
                    #     break

    param = None
    if mode == "batch_size":
        param = bs
    elif mode == "width":
        param = width
    elif mode == "depth":
        param = depth
    else:
        raise ValueError(f"Warning, invalid mode: {mode}")

    # if not os.path.exists(f"{ops_save_dir}/{model_name}"):
    #     os.makedirs(f"{ops_save_dir}/{model_name}")

    prof_type = "cpu" if record == ProfilerActivity.CPU else "cuda"
    df = prof_to_df(prof)
    df.to_csv(
        f"{ops_save_dir}/{model_name}/{model_name}-{prof_type}-{mode}-{param}.csv",
        index=False,
    )

    return running_loss


def main(
    node_feature_size,
    hidden_dim,
    num_conv_layers,
    num_heads,
    batch_size,
    record,
    mode,
    ops_save_dir,
    model_name,
    datatype,
):

    # model = GATREG(
    #     input_dim=node_feature_size,
    #     hidden_dim=hidden_dim,
    #     dropout=0,
    #     num_conv_layers=num_conv_layers,
    #     heads=num_heads,
    # )

    model = GraphTransformer(
        dim=256,
        depth=num_conv_layers,
        edge_dim=512,  # optional - if left out, edge dimensions is assumed to be the same as the node dimensions above
        with_feedforwards=True,  # whether to add a feedforward after each attention layer, suggested by literature to be needed
        gated_residual=True,  # to use the gated residual to prevent over-smoothing
        rel_pos_emb=False,  # set to True if the nodes are ordered, default to False
    )

    nodes = torch.randn(batch_size, 128, 256).to("cuda")
    edges = torch.randn(batch_size, 128, 128, 512).to("cuda")
    mask = torch.ones(batch_size, 128).bool().to("cuda")

    if datatype == "fp16":
        nodes = torch.randn(batch_size, 128, 256).to(torch.float16)
        edges = torch.randn(batch_size, 128, 128, 512).to(torch.float16)
        mask = torch.ones(batch_size, 128).bool().to(torch.float16)
        nodes = torch.randn(batch_size, 128, 256).to("cuda")
        edges = torch.randn(batch_size, 128, 128, 512).to("cuda")
        mask = torch.ones(batch_size, 128).bool().to("cuda")

    # nodes = torch.randn(1, 128, 256)
    # edges = torch.randn(1, 128, 128, 512)
    # mask = torch.ones(1, 128).bool()

    # nodes, edges = model(nodes, edges, mask=mask)
    if datatype == "fp32":
        print("Using fp32")
        model = model.to(torch.float32)

    elif datatype == "fp16":
        print("Using fp16")
        model = model.to(torch.float16)
    else:
        print(f"Data type {datatype} is invalid")

    model = model.to("cuda")

    params = sum(p.numel() for p in model.parameters())
    print(f"Num parameters: {params}")

    # import pdb

    # pdb.set_trace()

    # get dataset, splits
    dataset = PygPCQM4Mv2Dataset(root="./data", smiles2graph=smiles2graph)

    split_dict = dataset.get_idx_split()
    valid_idx = split_dict["valid"]

    valid_dataset = dataset[valid_idx]
    valid_dataset.data.edge_index = valid_dataset.data.edge_index.to("cuda")
    if datatype == "fp32":
        valid_dataset.data.edge_attr = valid_dataset.data.edge_attr.to(
            device="cuda", dtype=torch.float32
        )
        valid_dataset.data.x = valid_dataset.data.x.to(
            device="cuda", dtype=torch.float32
        )
        valid_dataset.data.y = valid_dataset.data.y.to(
            device="cuda", dtype=torch.float32
        )
    if datatype == "fp316":
        valid_dataset.data.edge_attr = valid_dataset.data.edge_attr.to(
            device="cuda", dtype=torch.float16
        )
        valid_dataset.data.x = valid_dataset.data.x.to(
            device="cuda", dtype=torch.float16
        )
        valid_dataset.data.y = valid_dataset.data.y.to(
            device="cuda", dtype=torch.float16
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

    valtime = None

    starttime = timeit.default_timer()
    va_loss = evaluate_epoch(
        va_loader,
        model,
        "cuda",
        bs=batch_size,
        width=hidden_dim,
        depth=num_conv_layers,
        record=record,
        mode=mode,
        ops_save_dir=ops_save_dir,
        model_name=model_name,
        nodes=nodes,
        edges=edges,
        mask=mask,
    )
    endtime = timeit.default_timer()
    valtime = endtime - starttime

    return valtime, params


if __name__ == "__main__":

    # Macros
    DATATYPE = "fp16"
    ARCH = "gTransformer"  # gTransformer
    MODE = "batch_size"  # batch_size, width, depth
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

            valtime, params = main(
                node_feature_size=9,
                hidden_dim=64,
                num_conv_layers=4,
                num_heads=32,
                batch_size=batchsize,
                record=RECORD,
                mode=MODE,
                ops_save_dir=OPS_SAVE_DIR,
                model_name=ARCH,
                datatype=DATATYPE,
            )

            # traintimes.append(traintime)
            valtimes.append(valtime)
            params_lst.append(params)

        results = {"val_times": valtimes, "params": params_lst}

        df = pd.DataFrame(results)
        df.to_csv(f"{LATENCY_SAVE_DIR}/A100/{ARCH}/{MODE}.csv")

    """
    Parameter width
    """
    if MODE == "width":
        possible_param_ws = [2 ** x for x in range(10)]

        cnt = 1

        for param_w in possible_param_ws:
            print(cnt)
            cnt += 1

            valtime, params = main(
                node_feature_size=9,
                hidden_dim=param_w,
                num_conv_layers=4,
                num_heads=32,
                batch_size=32,  # 2048
                record=RECORD,
                mode=MODE,
                ops_save_dir=OPS_SAVE_DIR,
                model_name=ARCH,
                datatype=DATATYPE,
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

        cnt = 1

        for param_l in possible_param_ls:
            print(cnt)
            cnt += 1

            valtime, params = main(
                node_feature_size=9,
                hidden_dim=64,
                num_conv_layers=param_l,
                num_heads=32,
                batch_size=32,  # 32
                record=RECORD,
                mode=MODE,
                ops_save_dir=OPS_SAVE_DIR,
                model_name=ARCH,
                datatype=DATATYPE,
            )
            # traintimes.append(traintime)
            valtimes.append(valtime)
            params_lst.append(params)

        results = {"val_times": valtimes, "params": params_lst}

        df = pd.DataFrame(results)
        df.to_csv(f"{LATENCY_SAVE_DIR}/A100/{ARCH}/{MODE}.csv")
