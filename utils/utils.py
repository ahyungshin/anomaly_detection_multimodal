import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


import random
import torch.backends.cudnn as cudnn

def seed_fix(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True 
    cudnn.enabled = False

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
