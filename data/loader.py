from torch.utils.data import DataLoader

from data.trajectories import TrajectoryDataset, seq_collate


def data_dset(args, path):
    dset = TrajectoryDataset(
        path,                    # 调用时只说了了一个dataset文件，没有别的子文件了。正确的。不然全拼到一起了。
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)
    return dset
def data_loader(args, dset, batch_size, shuffle=False):
    loader = DataLoader(
        dset,                   # custom dataset 
        batch_size=batch_size,
        shuffle=shuffle,        # 调用该函数时 没有输入shuffle，所以确实是false
        # num_workers=args.loader_num_workers,
        collate_fn=seq_collate,
        pin_memory=True)
    return loader

