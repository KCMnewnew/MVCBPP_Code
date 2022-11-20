import torch
from torch import nn
import numpy as np
if __name__ == '__main__':

    m = [[1,2,3,4],[5,6]]
    m_ = []
    for i in m:
        sum_w=0
        for j in i:
            sum_w+=j
        m_.append(sum_w)

    print(torch.mean(torch.tensor(m_,dtype=float)).item())