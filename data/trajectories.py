import logging
import os
import math
from IPython import embed
import numpy as np

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def seq_collate(data):
    (
        obs_seq_list,     # array (ped（一个scene的） ,pos,seq（限制obj_len）)   array组成的list，batch-size个array组成的list ， getitem获得一个array，dataloader内部处理成list
        pred_seq_list,
        obs_seq_rel_list,
        pred_seq_rel_list,
        non_linear_ped_list,
        loss_mask_list,
    ) = zip(*data) 

    _len = [len(seq) for seq in obs_seq_list]  # len dim 0 的数量。  得到list 每个元素（一共batch size 个元素）是一个scene的 人数 
    cum_start_idx = [0] + np.cumsum(_len).tolist() 
    seq_start_end = [
        [start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])
    ]
    # Data format: batch（ped）, input_size, seq_len
    # LSTM input format: seq_len, batch（ped）, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)      # 按ped （batch）维度和以前一样的叠加方法，再重排列
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)  # 记录解码方法。 【【】，【】，...】
    out = [
        obs_traj,      # 这里面每个元素就是batch 个 array 拼在一起。 
        pred_traj,
        obs_traj_rel,
        pred_traj_rel,
        non_linear_ped,
        loss_mask,
        seq_start_end,
    ]

    return tuple(out)     # tuple(out)


def read_file(_path, delim="\t"):
    data = []
    if delim == "tab":
        delim = "\t"
    elif delim == "space":
        delim = " "
    with open(_path, "r") as f:  # r read only #文件对象 f 可以迭代文件的内容，使您能够逐行读取文件，处理每一行的数据
        for line in f:
            line = line.strip().split(delim)     # strip() 方法用于去除文本行两端的空白字符，包括空格、制表符、换行符等 split（）eg：文本行 "John\t30\tNew York" 将被分割成 ["John", "30", "New York"]
            line = [float(i) for i in line]
            # line = [np.loadtxt(i,delimiter=',') for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
        self,
        data_dir,
        obs_len=8,
        pred_len=12,
        skip=1,
        threshold=0.002,
        min_ped=1,
        # delim=","
        delim="\t",
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)   # 返回字符串list，包含该目录下 所有文件和子目录的名称
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]  # 所有文件的路径
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:  # 字符串list中取一个字符串代表的文件地址来读。
            data = read_file(path, delim)  # dataset 中的一个txt文件，是N行4列的 数字 自定义函数独出来后转化为numpy array
            frames = np.unique(data[:, 0]).tolist() # data的第0列的所有行数据，即第0列代表frame 。以unique frame为 行array，最后构成list
            frame_data = []
            for frame in frames:                     # 保证按frame 为一组array， 塞进list（list中的每个元素对应同一个frame的数据）
                frame_data.append(data[frame == data[:, 0], :]) #debug确定 data是2维的。   # bool 运算，每个unique的frame 回去对应data的数据（2维），从而把frame_id 一样的行数据放到一起，整个frame_data 按照frame大小从前往后 【array1_fram1,array2_fram2,..】；注意目前行数还没有意义，
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip)) # frames的len代表有多少frame（时刻）
                    # N个球，M个连续球一组，有多少组的问题：减去最后一组，前面的所有球都可以是M球一组中第一个球，+1加上最后一组。/skip就是一步算还是多步一算。
            for idx in range(0, num_sequences * self.skip + 1, skip): ##？？ 为什么要+1 可能和ceil有关
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)  # list按idx取indices，就按frame数量取值，按行拼（行数增加）起来构成从t时刻到t+len时刻 的数据 array ，由于frame还在每一列中，所以行的意义还是没有意义。即每一行（frame，ped，x，y）只有frame符合要求（t，t_seq）的行组成2D array （N*4）
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1]) # dim1 是pedestrians 编号。 每一列一个意义（frame，ped，x，y）
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))   # 构成array （ped，pos，seq（frame））--由于前面的数据操作，实际还是对  每一行（frame，ped，x，y）的数据的组合 和堆叠，还没有变成真array（每个维度有每个维度的意义），所以这里的zeros创建的装填array的顺序没有要求一定相同
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))  #（ped，seq）
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq): # 同一个frame下对每个unique ped处理  
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]  # array n*4  （frame ，ped ，x，y）几行，存在的行只有 frame为指定值(t,t+seq)，ped为指定值 某个int。
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4) 
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx    # .index 在 frames 中第一次出现的索引位置
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1  # +1 是因为 取切片[a,a+seq]时，确实是seq这么长，但从index看，是a,..,a+seq-1, 所以头尾index差值会少1. 为了基数，得加一个
                    if pad_end - pad_front != self.seq_len:
                        continue                                          # 判断是否符合seq的frame长度要求。  ！！！ 这里有可能出现ped在 这一段连续时间 走出了检测范围，数据不完整。 这个人的数据就不要了//我的猜测，走着走着出去的人，多半在边缘。不影响中间的行人/（忽略了快速移动的人）
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])     # 截取 dim2 和dim3， 且列向量转化成行。 一行为x，一行为y ， 列代表 frame====》 开始凑出 （pos=2,seq=N）的array了。再增加一个维度就凑出（frame，pos，seq）了
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)  #array （pos=2,seq=N）
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]   # 从一列数据中获得相对值，错位减法。从 index =1  的列 即每个元素往前挪一位都是t+1， 而短了一个元素。为了凑齐，被减数也少一位数 （切片不包括切片结束位置 -1 所以少） 
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq  #array (ped,pos,seq) # dim1 直接从通过考虑的人开始加，  pad_front 总为0也合理，这样填充才对
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq   #  相对位置的第0个元素 是没有修改的，是0！   也合理，假设每个人都是从静止开始动。
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1 
                    num_peds_considered += 1

                if num_peds_considered >= min_ped:  # 一个scene中起码有1个人
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)  # 记录一个scene 有多少人， 后面可以通过这个解码数据，因为他把所有scene组成的array以此放进一个list后又concatenate了
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])    # 因为有的ped会不符合条件，所以scene的ped数量可能变化，所以不用peds_in_curr_seq 来做切片
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)      #array (ped 堆叠,pos,seq)  但只要直到每个scene的ped_size（num_peds_in_seq） 就能还原出来      # 对list 经行concate。 list中含有多个ndarray (ped,pos,seq)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(seq_list[:, :, : self.obs_len]).type(
            torch.float
        )
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len :]).type(
            torch.float
        )
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, : self.obs_len]).type(
            torch.float
        )
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len :]).type(
            torch.float
        )
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()        # 逐级累加 由于0开始 之前scene的总人数，就是下一个scene的起始点（纳入的切片起点）         # num_peds_in_seq 记录了每个scene的人数。用于对dim0 取切片从而还原每个scene的array
        self.seq_start_end = [
            (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        # print("seq_start_end")

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):  # info in one batch.
        start, end = self.seq_start_end[index]  # trajectories num in one scene  (1,5) => from 1 to 5 is in the same scene
        out = [
            self.obs_traj[start:end, :],     # array (ped（一个scene的） ,pos,seq（限制obj_len）)
            self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :],
            self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end],
            self.loss_mask[start:end, :],
        ]
        return out
