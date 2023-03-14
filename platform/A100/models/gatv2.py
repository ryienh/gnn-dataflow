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
from torch.profiler import profile, record_function, ProfilerActivity

from util import prof_to_df


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


def evaluate_epoch(val_loader, model, device, bs, width, depth, record):

    model = model.eval()

    running_loss = 0
    predictions = []
    labels = []

    cntr = 1
    with torch.no_grad():

        for X in tqdm.tqdm(val_loader):
            # X = X.to(device)
            X.y = X.y.to(torch.float32)

            logits = None

            with profile(
                activities=[record],
                record_shapes=True,
            ) as prof:
                with record_function("model_inference"):
                    # model(inputs)
                    logits = model(X)

            cntr += 1

            if cntr == 10:
                break

            prediction = torch.squeeze(logits)
            my_loss = torch.nn.functional.mse_loss(prediction, X.y)

            # loss calculation
            running_loss += my_loss.item() * X.num_graphs

            try:
                predictions += prediction.tolist()
            except Exception:
                pass
            labels += X.y.tolist()

        running_loss /= len(val_loader.dataset)

    df = prof_to_df(prof)
    df.to_csv(
        f"/lus/grand/projects/datascience/gnn-dataflow/profiling_data/gtransformer/gtransformer_width_{width}.csv",
        index=False,
    )

    return running_loss


def main(node_feature_size, hidden_dim, num_conv_layers, num_heads, batch_size, record):

    model = GATREG(
        input_dim=node_feature_size,
        hidden_dim=hidden_dim,
        dropout=0,
        num_conv_layers=num_conv_layers,
        heads=num_heads,
    )
    model = model.to(torch.float32)
    model = model.to("cuda")

    params = sum(p.numel() for p in model.parameters())
    print(f"Num parameters: {params}")

    # get dataset, splits
    dataset = PygPCQM4Mv2Dataset(root="./data", smiles2graph=smiles2graph)

    split_dict = dataset.get_idx_split()
    valid_idx = split_dict["valid"]

    valid_dataset = dataset[valid_idx]
    valid_dataset = valid_dataset.data.to("cuda")

    va_loader = DataLoader(
        dataset=dataset[valid_idx],
        batch_size=batch_size,
        shuffle=False,
    )

    traintime = None
    valtime = None

    for epoch in range(0, 2):

        if epoch == 1:

            starttime = timeit.default_timer()
            va_loss = evaluate_epoch(
                va_loader,
                model,
                "cuda",
                bs=batch_size,
                width=hidden_dim,
                depth=num_conv_layers,
                record=record,
            )
            endtime = timeit.default_timer()
            valtime = endtime - starttime

    return traintime, valtime, params


if __name__ == "__main__":

    # Macros
    ARCH = "GATv2"  # gTransformer
    MODE = "batch_size"  # , width, depth
    OPS_SAVE_DIR = "/lus/grand/projects/datascience/gnn-dataflow/profiling_data"
    LATENCY_SAVE_DIR = "./logs"
    RECORD = ProfilerActivity.CPU  # ProfilerActivity.CUDA

    torch.manual_seed(0)

    valtimes = []
    params_lst = []

    """
    Batch size
    """
    if MODE == "batch_size":
        batchsizes = [2 ** x for x in range(2, 14)]
        for batchsize in batchsizes:  # 10, 12000

            _, valtime, params = main(
                node_feature_size=9,
                hidden_dim=64,
                num_conv_layers=4,
                num_heads=32,
                batch_size=batchsize,
                record=RECORD,
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

            _, valtime, params = main(
                node_feature_size=9,
                hidden_dim=param_w,
                num_conv_layers=4,
                num_heads=32,
                batch_size=2048,
                record=RECORD,
            )

            # traintimes.append(traintime)
            valtimes.append(valtime)
            params_lst.append(params)

        results = {"val_times": valtimes, "params": params_lst}

        df = pd.DataFrame(results)
        df.to_csv(f"{LATENCY_SAVE_DIR}/A100/{ARCH}/{MODE}.csv")

    """
    Parameter length
    """
    if MODE == "length":
        possible_param_ls = [2 ** x for x in range(10)]

        cnt = 1

        for param_l in possible_param_ls:
            print(cnt)
            cnt += 1

            _, valtime, params = main(
                node_feature_size=9,
                hidden_dim=64,
                num_conv_layers=param_l,
                num_heads=32,
                batch_size=2048,
                record=RECORD,
            )

            # traintimes.append(traintime)
            valtimes.append(valtime)
            params_lst.append(params)

        results = {"val_times": valtimes, "params": params_lst}

        df = pd.DataFrame(results)
        df.to_csv(f"{LATENCY_SAVE_DIR}/A100/{ARCH}/{MODE}.csv")
