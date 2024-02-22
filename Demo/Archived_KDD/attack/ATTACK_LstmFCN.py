import sys
import os


def parent_dir(path, n):
    for _ in range(n):
        path = os.path.dirname(path)
    return path


sys.path.append(parent_dir(__file__, 4))
from __init__ import *

from CODE.Train.lstm_fcn import *

train_model_name = LSTMFCN
# torch.backends.cudnn.enabled = False
ATTACK_ALL(
    train_model_name,
    run_time=5,
)
