import torch
import numpy as np
def to_one_hot(x,gate_size,device): # gate_input convert into one-hot. from(batch_size,1) into (batch_size, gate_size)
    # task id start from 1, not 0
    if type(x)==torch.Tensor:
        device=x.device
        x = x.cpu()
    temp = np.zeros(shape=[len(x),gate_size],dtype='float32')
    temp[range(len(x)),x-1]=1. # x is maybe [1,1,1] start from 1, but in array should occupy index 0 col
    temp = torch.from_numpy(temp)
    return temp if device is None else temp.to(device)
    