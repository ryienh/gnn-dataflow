import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# plt.style.use("ipynb")


def main(dim):
    """
    Batch size
    """
    if dim == "batch_size":
        # load data
        df = pd.read_csv("logs/fp32/A100/GATv2/batch_size.csv")

        val_time = df["val_times"].to_numpy()
        params = df["params"].to_numpy()

        val_time = val_time[1:]

        # batch_size = range(260, 12000, 250)
        batch_size = [2 ** x for x in range(1, 14)]
        batch_size = batch_size[:-1]

        plt.plot(batch_size, val_time)
        plt.xlabel("Batch size")
        plt.ylabel("Inference time (s)")
        plt.title("GATv2 Batch size vs. Inference time. Params=558977")
        # plt.yscale("log")
        plt.xscale("log")
        plt.savefig("temp/batch_size_gat.png")

    """
    Width
    """
    if dim == "width":
        df = pd.read_csv("logs/fp32/A100/GATv2/width.csv")

        val_time = df["val_times"].to_numpy()
        params = df["params"].to_numpy()

        # val_time = val_time[1:]

        width = [2 ** x for x in range(10)]

        plt.plot(params, val_time)
        plt.xlabel("Num Params")
        plt.ylabel("Inference time (s)")
        plt.title("GATv2 Width vs. Inference time. Batch size = 2048")
        # plt.yscale("log")
        plt.xscale("log")
        plt.savefig("temp/width_gat.png")

    """
    Length
    """
    if dim == "depth":
        df = pd.read_csv("logs/fp32/A100/gTransformer/depth.csv")

        val_time = df["val_times"].to_numpy()
        params = df["params"].to_numpy()

        # val_time = val_time[1:]

        length = [2 ** x for x in range(10)]

        plt.plot(params, val_time)
        plt.xlabel("Num Params")
        plt.ylabel("Inference time (s)")
        plt.title("Graph Transformer Num Layers vs. Inference time. Batch size = 2048")
        # plt.yscale("log")
        plt.xscale("log")
        plt.savefig("temp/depth_gt.png")

    """
    Length
    """
    if dim == "hidden_channels":
        df = pd.read_csv("logs/fp32/A100/schnet/num_interactions.csv")

        val_time = df["val_times"].to_numpy()
        params = df["params"].to_numpy()

        # val_time = val_time[1:]

        # length = [2 ** x for x in range(10)]

        plt.plot(params, val_time)
        plt.xlabel("Num Params")
        plt.ylabel("Inference time (s)")
        plt.title("Schnet Batch Size vs. Inference time. Batch size = 2048")
        # plt.yscale("log")
        plt.xscale("log")
        plt.savefig("temp/batch_size_schnet.png")


if __name__ == "__main__":
    # DIM = "width"
    # DIM = "depth"
    DIM = "hidden_channels"
    main(dim=DIM)
