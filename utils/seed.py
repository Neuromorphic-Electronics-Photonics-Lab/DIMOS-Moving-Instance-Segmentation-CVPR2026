import random
import numpy as np
import torch


def set_seed(seed=1234, thread=1, gpu=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.set_num_threads(thread)


