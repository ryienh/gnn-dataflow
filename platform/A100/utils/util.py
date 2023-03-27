"""
Collection of utility functions for training and evaluation scripts.
"""
import os
import torch
import itertools
import random
import numpy as np
import pandas as pd


def get_config_tgn(attr, fname=os.path.join("..", "configs", "config_tgn.json")):
    """
    Retrieves the queried attribute value from the config file. Loads the
    config file on first call.

    Parameters
    ----------
    attr : str
        Size of train+val+test sample
    fname : os.path, optional
        Path to config file, default is "./config.json"

    Returns
    -------
    Requested attribute
    """
    if not hasattr(get_config, "config"):
        with open(fname) as f:
            get_config.config = eval(f.read())
    node = get_config.config
    for part in attr.split("."):
        node = node[part]
    return node


def get_config(attr, fname=os.path.join("./", "config.json")):
    """
    Retrieves the queried attribute value from the config file. Loads the
    config file on first call.

    Parameters
    ----------
    attr : str
        Size of train+val+test sample
    fname : os.path, optional
        Path to config file, default is "./config.json"

    Returns
    -------
    Requested attribute
    """
    if not hasattr(get_config, "config"):
        with open(fname) as f:
            get_config.config = eval(f.read())
    node = get_config.config
    for part in attr.split("."):
        node = node[part]
    return node


def preprocess(smp, device):
    # add start of sequence
    sos = torch.ones(smp.shape[:-1] + ((1,)), device=device)
    smp = torch.cat((sos, smp), -1)
    return smp.unsqueeze(-1)


def subsample_dst(
    pos_dst,
    rand_all_dst,
    n_sampled_dst,
    rand_smpdst,
    all_dst,
    min_dst_idx,
    tmap,
    device,
    sample_all=False,
):
    """
    Returns:
    `sampled_dst` size: (n_sampled_dst).
        contains all unique pos_dst (correct destinations) and other random (wrong) destinations
    `idx_pos_dst` size: (batch_size).
        element i contains the index of the positive destination of source i in sampled_dst
    """
    if sample_all:
        return all_dst, pos_dst - min_dst_idx
    else:
        pos_dst_u = pos_dst.unique()
        rand_all_dst.random_()
        # make sure all pos_dst_u are included
        # (put negative values on the indexes of the pos_dst so that they are first when applying topk)
        rand_all_dst[pos_dst_u - min_dst_idx] = (
            -torch.arange(len(pos_dst_u), 0, -1).float().to(device)
        )
        # select a random sample
        _, rand_perm = rand_all_dst.topk(
            n_sampled_dst, dim=0, sorted=True, largest=False
        )
        # shuffle again
        rand_smpdst.random_()
        _, r = rand_smpdst.topk(n_sampled_dst, dim=0, sorted=True, largest=False)
        sampled_dst = all_dst[torch.scatter(rand_perm, dim=0, index=r, src=rand_perm)]
        # map pos_dst to the right indexes
        idx_pos_dst_u = r[
            : len(pos_dst_u)
        ]  # indexes of the correct destinations in `sampled_dst`
        tmap[
            pos_dst_u
        ] = idx_pos_dst_u  # pos_dst original index -> corresponding index in `sampled_dst`
        idx_pos_dst = tmap[
            pos_dst
        ]  # translate the pos_dst original indexes to the new indexes in `sampled_dst`
        return sampled_dst, idx_pos_dst


def subsample_src(
    pos_src,
    all_src,
    rand_all_src,
    rand_smpsrc,
    tmap_src,
    n_sampled_src,
    device,
    sample_all=False,
):
    """
    Returns:
    `sampled_dst` size: (n_sampled_dst).
        contains all unique pos_dst (correct destinations) and other random (wrong) destinations
    `idx_pos_dst` size: (batch_size).
        element i contains the index of the positive destination of source i in sampled_dst
    """
    if sample_all:
        return all_src, pos_src
    else:
        pos_src_u = pos_src.unique()
        rand_all_src.random_()
        # make sure all pos_src_u are included
        # (put negative values on the indexes of the pos_src so that they are first when applying topk)
        rand_all_src[pos_src_u] = (
            -torch.arange(len(pos_src_u), 0, -1).float().to(device)
        )
        # select a random sample
        _, rand_perm = rand_all_src.topk(
            n_sampled_src, dim=0, sorted=True, largest=False
        )
        # shuffle again
        rand_smpsrc.random_()
        _, r = rand_smpsrc.topk(n_sampled_src, dim=0, sorted=True, largest=False)
        sampled_src = all_src[torch.scatter(rand_perm, dim=0, index=r, src=rand_perm)]
        # map pos_dst to the right indexes
        idx_pos_src_u = r[
            : len(pos_src_u)
        ]  # indexes of the correct destinations in `sampled_dst`
        tmap_src[
            pos_src_u
        ] = idx_pos_src_u  # pos_dst original index -> corresponding index in `sampled_dst`
        idx_pos_src = tmap_src[
            pos_src
        ]  # translate the pos_dst original indexes to the new indexes in `sampled_dst`
        return sampled_src, idx_pos_src


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, epoch, checkpoint_dir):
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
    }

    filename = os.path.join(checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(epoch))
    torch.save(state, filename)


def restore_checkpoint(model, checkpoint_dir, cuda=True, force=False, pretrain=False):
    """
    If a checkpoint exists, restores the PyTorch model from the checkpoint.
    Returns the model and the current epoch.
    """
    files = [
        fn
        for fn in os.listdir(checkpoint_dir)
        if fn.startswith("epoch=") and fn.endswith(".checkpoint.pth.tar")
    ]

    if not files:
        print("No saved models found")
        if force:
            raise Exception("Checkpoint not found")
        else:
            return model, 0

    # Find latest epoch
    for i in itertools.count(1):
        if "epoch={}.checkpoint.pth.tar".format(i) in files:
            epoch = i
        else:
            break

    if not force:
        print(
            f"Select epoch: Choose in range [0, {epoch}].",
            "Entering 0 will train from scratch.",
        )
        print(">> ", end="")
        in_epoch = int(input())
        if in_epoch not in range(epoch + 1):
            raise Exception("Invalid epoch number")
        if in_epoch == 0:
            print("Checkpoint not loaded")
            clear_checkpoint(checkpoint_dir)
            return model, 0
    else:
        print(f"Select epoch: Choose in range [1, {epoch}].")
        in_epoch = int(input())
        if in_epoch not in range(1, epoch + 1):
            raise Exception("Invalid epoch number")

    filename = os.path.join(checkpoint_dir, f"epoch={in_epoch}.checkpoint.pth.tar")

    print("Loading from checkpoint {}?".format(filename))

    if cuda:
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)

    try:
        if pretrain:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint["state_dict"])
        print(
            "=> Successfully restored checkpoint (trained for {} epochs)".format(
                checkpoint["epoch"]
            )
        )
    except:
        print("=> Checkpoint not successfully restored")
        raise

    return model, in_epoch


def clear_checkpoint(checkpoint_dir):
    fnames = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")]
    for f in fnames:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint removed")


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return


def train_val1_val2_test_split(
    data, val_1_ratio: float = 0.10, val_2_ratio: float = 0.10, test_ratio: float = 0.10
):
    r"""Splits the data in training, validation and test sets according to
    time.

    Args:
        val_ratio (float, optional): The proportion (in percents) of the
            dataset to include in the validation split.
            (default: :obj:`0.15`)
        test_ratio (float, optional): The proportion (in percents) of the
            dataset to include in the test split. (default: :obj:`0.15`)
    """
    val_1_time, val_2_time, test_time = np.quantile(
        data.t.cpu().numpy(),
        [
            1.0 - val_1_ratio - val_2_ratio - test_ratio,
            1.0 - val_2_ratio - test_ratio,
            1.0 - test_ratio,
        ],
    )

    val_1_idx = int((data.t <= val_1_time).sum())
    val_2_idx = int((data.t <= val_2_time).sum())
    test_idx = int((data.t <= test_time).sum())

    return (
        data[:val_1_idx],
        data[val_1_idx:val_2_idx],
        data[val_2_idx:test_idx],
        data[test_idx:],
    )


def save_model(best_model, path, config_dict, neighbor_loader):

    torch.save(
        {
            "memory_state_dict": best_model[0].state_dict(),
            "gnn_state_dict": best_model[1].state_dict(),
            "embd_to_score_dst_state_dict": best_model[2].state_dict(),
            "embd_to_score_src_state_dict": best_model[3].state_dict(),
            "feats_model_state_dict": best_model[4].state_dict(),
            "embd_to_h0_state_dict": best_model[5].state_dict(),
            "config_dict": config_dict,
            "neighbor_loader": neighbor_loader,
        },
        path,
    )


def save_tgn():
    pass


def load_tgn():
    pass


UID = "uid"

def get_children(df, op_id):
#     return df[df["id"].isin(df[df["id"] == op_id]["cpu_children"].tolist()[0])]
    return df[df[UID].isin(np.concatenate(df[df[UID] == op_id]["cpu_children"].to_numpy()))]


def prof_to_df(prof):
    """
    Extract info from `torch.profiler.profiler.profile` object `prof` into a `pandas.DataFrame`
    
    Example:
    
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, 
                            torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
                ) as prof:
            
            # [code to profile here]
            
        df = prof_to_df(prof)
        df.to_csv("/lus/eagle/projects/datascience/gnn-dataflow/profiling_data/model_XYZ_profile.csv", index=False)
    """
#     df = pd.DataFrame.from_dict([e.__dict__ for e in prof.events()])
    df = pd.DataFrame.from_dict([{**{"FE": e}, **e.__dict__} for e in prof.events()])
    df[UID] = range(len(df))
    df["time_range"] = df["time_range"].apply(lambda x: x.elapsed_us())
    #df["cpu_children"] = df["cpu_children"].apply(lambda x: [c.id for c in x])
    #df["cpu_children"] = df["cpu_children"].apply(lambda x: [df[df["FE"].__eq__(c)][UID].item() for c in x])
    UID2FE = df.set_index(UID)["FE"].to_dict()
    id2UID_or_dfFE = {k: df.iloc[v[0]][UID] if len(v) == 1 else pd.DataFrame([df.iloc[i] for i in v]) \
        for k,v in df.groupby("id").groups.items()}
    def get_child_uid(c, df):
        return id2UID_or_dfFE[c.id] if not isinstance(id2UID_or_dfFE[c.id], pd.DataFrame) \
                                    and UID2FE[id2UID_or_dfFE[c.id]].__eq__(c) \
            else id2UID_or_dfFE[c.id][id2UID_or_dfFE[c.id]["FE"].__eq__(c)][UID].item()
    df["cpu_children"] = df["cpu_children"].apply(lambda x: [get_child_uid(c, df) for c in x])  
    
    #df["cpu_parent"] = df["cpu_parent"].apply(lambda x: x.id if x is not None else -1)
    #df["cpu_parent"] = df["cpu_parent"].apply(lambda x: df[df["FE"].__eq__(x)][UID].item() if x is not None else -1)
    df["cpu_parent"] = df["cpu_parent"].apply(lambda x: get_child_uid(x, df) if x is not None else -1)
    
    df["input_shapes"] = df["input_shapes"].astype(str)
    # get operator self time
    #df["children_time"] = df.apply(lambda x: get_children(df, x[UID])["time_range"].sum(), axis=1)
    uid2childern_uids = df.set_index(UID)["cpu_children"].to_dict()
    df["children_time"] = df.apply(lambda x: df[df[UID].isin(uid2childern_uids[x[UID]])]["time_range"].sum(), axis=1)
    
    df["self_time"] = df["time_range"] - df["children_time"]
    total_time = df["self_time"].sum()
    df["percent_self_time"] = 100 * df["self_time"] / total_time
    return df.drop(columns=["FE"])

