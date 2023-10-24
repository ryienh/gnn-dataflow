"""
Adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
"""
import click
import os.path as osp
import timeit
import pandas as pd
import tqdm

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from util import prof_to_df

from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)


def validate(
    batch_size=200,
    memory_dim=100,
    embedding_dim=100,
    time_dim=100,
    record=None,
    mode=None,
    ops_save_dir=None,
    profiler_dir=None,
    datatype=None,
    device=None,
    seed=None
):

    if datatype != "fp32":
        raise ValueError("Sorry, profiling only implemented for fp32 at this time.")

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'JODIE')
    dataset = JODIEDataset(path, name='wikipedia')
    data = dataset[0]

    # For small datasets, we can put the whole dataset on GPU and thus avoid
    # expensive memory transfer costs for mini-batches:
    data = data.to(device)

    train_data, val_data, test_data = data.train_val_test_split(
        val_ratio=0.15, test_ratio=0.15)

    train_loader = TemporalDataLoader(
        train_data,
        batch_size=batch_size,
        neg_sampling_ratio=1.0,
    )
    val_loader = TemporalDataLoader(
        val_data,
        batch_size=batch_size,
        neg_sampling_ratio=1.0,
    )
    neighbor_loader = LastNeighborLoader(data.num_nodes, size=10, device=device)

    class GraphAttentionEmbedding(torch.nn.Module):
        def __init__(self, in_channels, out_channels, msg_dim, time_enc):
            super().__init__()
            self.time_enc = time_enc
            edge_dim = msg_dim + time_enc.out_channels
            self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                        dropout=0.1, edge_dim=edge_dim)

        def forward(self, x, last_update, edge_index, t, msg):
            rel_t = last_update[edge_index[0]] - t
            rel_t_enc = self.time_enc(rel_t.to(x.dtype))
            edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
            return self.conv(x, edge_index, edge_attr)

    class LinkPredictor(torch.nn.Module):
        def __init__(self, in_channels):
            super().__init__()
            self.lin_src = Linear(in_channels, in_channels)
            self.lin_dst = Linear(in_channels, in_channels)
            self.lin_final = Linear(in_channels, 1)

        def forward(self, z_src, z_dst):
            h = self.lin_src(z_src) + self.lin_dst(z_dst)
            h = h.relu()
            return self.lin_final(h)

    memory = TGNMemory(
        data.num_nodes,
        data.msg.size(-1),
        memory_dim,
        time_dim,
        message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=memory_dim,
        out_channels=embedding_dim,
        msg_dim=data.msg.size(-1),
        time_enc=memory.time_enc,
    ).to(device)

    link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters())
        | set(link_pred.parameters()), lr=0.0001)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Helper vector to map global node indices to local ones.
    assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

    def train():
        memory.train()
        gnn.train()
        link_pred.train()

        memory.reset_state()  # Start with a fresh memory.
        neighbor_loader.reset_state()  # Start with an empty graph.

        total_loss = 0
        print("Begin one epoch of tgn training")
        for batch in tqdm.tqdm(train_loader):

            # Not necessary to do full epoch of training
            # if idx >= 9:
            #     break

            optimizer.zero_grad()
            batch = batch.to(device)

            n_id, edge_index, e_id = neighbor_loader(batch.n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            # Get updated memory of all nodes involved in the computation.
            z, last_update = memory(n_id)
            z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                    data.msg[e_id].to(device))
            pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
            neg_out = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])

            loss = criterion(pos_out, torch.ones_like(pos_out))
            loss += criterion(neg_out, torch.zeros_like(neg_out))

            # Update memory and neighbor loader with ground-truth state.
            memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            neighbor_loader.insert(batch.src, batch.dst)

            loss.backward()
            optimizer.step()
            memory.detach()
            total_loss += float(loss) * batch.num_events

        print("End one epoch of tgn training")

        return total_loss / train_data.num_events

    @torch.no_grad()
    def test(loader):

        # import pdb
        # pdb.set_trace()

        memory.eval()
        gnn.eval()
        link_pred.eval()

        torch.manual_seed(seed)  # Ensure deterministic sampling across epochs.

        memory_params = sum(p.numel() for p in memory.parameters())
        gnn_params = sum(p.numel() for p in gnn.parameters())
        link_pred_params = sum(p.numel() for p in link_pred.parameters())
        params = memory_params + gnn_params + link_pred_params
        print(f"Num parameters: {params}")

        my_schedule = schedule(wait=5, warmup=1, active=4)  # repeat=1

        aps, aucs = [], []

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

                    for step, data in enumerate(loader):

                        if step >= 4:  # FIXME: ask filippo about this
                            break

                        starttime = timeit.default_timer()

                        batch = data.to(device)

                        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
                        assoc[n_id] = torch.arange(n_id.size(0), device=device)

                        z, last_update = memory(n_id)
                        z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                                data.msg[e_id].to(device))
                        pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
                        neg_out = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])

                        y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
                        y_true = torch.cat(
                            [torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0)

                        aps.append(average_precision_score(y_true, y_pred))
                        aucs.append(roc_auc_score(y_true, y_pred))

                        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
                        neighbor_loader.insert(batch.src, batch.dst)

                        endtime = timeit.default_timer()
                        valtimes.append(endtime - starttime)
                        valtime_count += 1

                        prof.step()

                    param = None
                    if mode == "batch_size":
                        param = batch_size
                    elif mode == "memory_dim":
                        param = memory_dim
                    elif mode == "embedding_dim":
                        param = embedding_dim
                    elif mode == "time_dim":
                        param = time_dim
                    else:
                        raise ValueError(f"Warning, invalid mode: {mode}")

                    prof_type = "cpu" if record == ProfilerActivity.CPU else "cuda"
                    df = prof_to_df(prof)
                    df.to_csv(
                        f"{ops_save_dir}/tgn/tgn-{prof_type}-{mode}-{param}.csv",
                        index=False,
                    )

        valtime = sum(valtimes)/valtime_count
        return valtime, params

    for epoch in range(1, 2):
        loss = train()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
        valtime, params = test(val_loader)
        return valtime, params


@click.command()
@click.option('--datatype', default="fp32", help='Percision used for data and model weights. One of "fp32" or "fp16".')
@click.option('--mode', default=None, help='Mode used for benchmarking. One of "batch_size", "memory_dim", "embedding_dim", "time_dim", or "all".')
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

    if MODE == "all":
        print(f"Running tgn in all modes: 'batch_size', 'width', 'depth'")

    """
    Batch size
    """
    if MODE in ["all", "batch_size"]:
        batchsizes = [2 ** x for x in range(1, 14)]  # TODO: tune
        for idx, batchsize in enumerate(batchsizes):

            print(f"Iteration: {idx+1}, batchsize: {batchsize}")

            valtime, params = validate(
                batch_size=batchsize,
                record=RECORD,
                mode=MODE,
                ops_save_dir=OPS_SAVE_DIR,
                profiler_dir=PROFILER_DIR,
                datatype=DATATYPE,
                device=DEVICE,
                seed=seed
            )  # type: ignore

            valtimes.append(valtime)
            params_lst.append(params)
            batch_size_lst.append(batchsize)

        results = {"val_times": valtimes, "params": params_lst, "batch_size": batch_size_lst}

        df = pd.DataFrame(results)
        df.to_csv(f"{LATENCY_SAVE_DIR}/A100/tgn/{MODE}.csv")

    """
    Memory dim
    """
    valtimes = []
    params_lst = []
    batch_size_lst = []

    if MODE in ["all", "memory_dim"]:
        possible_param_mem = [2 ** x for x in range(0, 16)]  # TODO: tune

        for idx, param_mem in enumerate(possible_param_mem):

            print(f"Iteration: {idx+1}, param_w: {param_mem}")

            valtime, params = validate(
                memory_dim=param_mem,
                record=RECORD,
                mode=MODE,
                ops_save_dir=OPS_SAVE_DIR,
                profiler_dir=PROFILER_DIR,
                datatype=DATATYPE,
                device=DEVICE,
                seed=seed
            )  # type: ignore

            valtimes.append(valtime)
            params_lst.append(params)
            batch_size_lst.append(200)

        results = {"val_times": valtimes, "params": params_lst, "batch_size": batch_size_lst}

        df = pd.DataFrame(results)
        df.to_csv(f"{LATENCY_SAVE_DIR}/A100/tgn/{MODE}.csv")

    """
    Embedding dim
    """
    valtimes = []
    params_lst = []
    batch_size_lst = []

    if MODE in ["all", "embedding_dim"]:
        possible_param_emb = [2 ** x for x in range(0, 40)]  # TODO: tune

        for idx, param_emb in enumerate(possible_param_emb):

            print(f"Iteration: {idx+1} param_f: {param_emb}")

            valtime, params = validate(
                embedding_dim=param_emb,
                record=RECORD,
                mode=MODE,
                ops_save_dir=OPS_SAVE_DIR,
                profiler_dir=PROFILER_DIR,
                datatype=DATATYPE,
                device=DEVICE,
                seed=seed
            )  # type: ignore

            valtimes.append(valtime)
            params_lst.append(params)
            batch_size_lst.append(200)

        results = {"val_times": valtimes, "params": params_lst, "batch_size": batch_size_lst}

        df = pd.DataFrame(results)
        df.to_csv(f"{LATENCY_SAVE_DIR}/A100/tgn/{MODE}.csv")

    """
    Time dim
    """
    valtimes = []
    params_lst = []
    batch_size_lst = []

    if MODE in ["all", "time_dim"]:
        # LINEAR
        possible_param_time = [2 ** x for x in range(1, 26)]  # TODO: tune

        for idx, param_t in enumerate(possible_param_time):

            print(f"Iteration: {idx+1} param_i: {param_t}")

            valtime, params = validate(
                time_dim=param_t,
                record=RECORD,
                mode=MODE,
                ops_save_dir=OPS_SAVE_DIR,
                profiler_dir=PROFILER_DIR,
                datatype=DATATYPE,
                device=DEVICE,
                seed=seed
            )  # type: ignore

            valtimes.append(valtime)
            params_lst.append(params)
            batch_size_lst.append(200)

        results = {"val_times": valtimes, "params": params_lst, "batch_size": batch_size_lst}

        df = pd.DataFrame(results)
        df.to_csv(f"{LATENCY_SAVE_DIR}/A100/tgn/{MODE}.csv")


if __name__ == "__main__":
    cli()
