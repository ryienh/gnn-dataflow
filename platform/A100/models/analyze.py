import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# plt.style.use("ipynb")

def main(dim):
    """
    Batch size
    """
    if dim == "batch_size"
        # load data
        df = pd.read_csv("temp/batch_size.csv")

        val_time = df["val_times"].to_numpy()
        params = df["params"].to_numpy()

        val_time = val_time[1:]

        # batch_size = range(260, 12000, 250)
        batch_size = [2 ** x for x in range(2, 14)]
        batch_size = batch_size[:-1]

        plt.plot(batch_size, val_time)
        plt.xlabel("Batch size")
        plt.ylabel("Inference time (s)")
        plt.title("Graph Transformer Batch size vs. Inference time. Params=558977")
        # plt.yscale("log")
        plt.xscale("log")
        plt.savefig("temp/batch_size_gt.png")

    """
    Width
    """
    if dim == "width":
        df = pd.read_csv("temp/width.csv")

        val_time = df["val_times"].to_numpy()
        params = df["params"].to_numpy()

        # val_time = val_time[1:]

        width = [2 ** x for x in range(10)]

        plt.plot(params, val_time)
        plt.xlabel("Num Params")
        plt.ylabel("Inference time (s)")
        plt.title("Graph Transformer Width vs. Inference time. Batch size = 2048")
        # plt.yscale("log")
        plt.xscale("log")
        plt.savefig("temp/width_gt.png")


    """
    Length
    """
    if dim == "length":
        df = pd.read_csv("temp/length.csv")

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
        plt.savefig("temp/length_gt.png")

if __name__ == "__main__":
    DIM = "batch_size"
    main(dim=DIM)