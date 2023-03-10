import torch
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.loader import DataLoader
from ogb.utils import smiles2graph
from ogb.lsc import PygPCQM4Mv2Dataset, PCQM4Mv2Evaluator
import tqdm
from deepspeed.profiling.flops_profiler import FlopsProfiler
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity


# model definition
class GraphTransformer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, num_conv_layers, heads):

        super(GraphTransformer, self).__init__()

        self.dropout = dropout
        self.num_layers = num_conv_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim, heads))
        self.lns = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim, heads))
            self.lns.append(
                torch.nn.LayerNorm(hidden_dim)
            )  # one less lns than conv bc no lns after final conv

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
        if data.num_node_features == 0:
            print("Warning: No node features detected.")
            x = torch.ones(data.num_nodes, 1)

        x = x.to(torch.float32)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            # emb = x
            x = self.ReLU(x)
            x = self.conv_dropout(x)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        # pooling
        x = global_mean_pool(x, batch)

        # MLP
        x = self.post_mp(x)

        return x


# get data
pyg_dataset = PygPCQM4Mv2Dataset(root="../../data/PCQM4Mv2", smiles2graph=smiles2graph)
split_dict = pyg_dataset.get_idx_split()
train_idx = split_dict["train"]  # numpy array storing indices of training molecules
data = pyg_dataset[train_idx]

model = GraphTransformer(
    input_dim=9, hidden_dim=256, dropout=0.0, num_conv_layers=10, heads=32
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
device = "cuda"
tr_loader = DataLoader(data, batch_size=512, shuffle=True)
model = model.to(device)

for epoch in range(1):

    model = model.train()
    running_loss = 0
    predictions = []
    labels = []

    profile_step = 5
    # prof = FlopsProfiler(model)
    print_profile = True

    for step, X in enumerate(tr_loader):

        # if step == profile_step:
        #     prof.start_profile()

        X = X.to(device)

        # clear parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize

        # prediction = model(X)
        prediction = None

        if step == profile_step:  # if using multi nodes, check global_rank == 0 as well
            #     prof.stop_profile()
            #     flops = prof.get_total_flops()
            #     macs = prof.get_total_macs()
            #     params = prof.get_total_params()
            #     if print_profile:
            #         prof.print_model_profile(profile_step=profile_step)
            #     prof.end_profile()

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
            ) as prof:
                with record_function("model_train"):
                    prediction = model(X)

            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))

        else:
            prediction = model(X)

        prediction = torch.squeeze(prediction)
        loss = torch.nn.functional.l1_loss(prediction, X.y)
        loss.backward()
        optimizer.step()

        predictions += prediction.tolist()
        labels += X.y.tolist()

        # calculate loss
        running_loss += loss.item() * X.num_graphs

    running_loss /= len(tr_loader.dataset)

    # train_mae = mean_absolute_error(y_true=labels, y_pred=predictions)
