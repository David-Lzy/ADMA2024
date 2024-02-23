import sys
import os


def parent_dir(path, n):
    for _ in range(n):
        path = os.path.dirname(path)
    return path


sys.path.append(parent_dir(__file__, 3))
from __init__ import *

logging.getLogger().setLevel(logging.INFO)

from CODE.Train.inception_time import *

train_model_name = Classifier_INCEPTION


ATTACK_ALL(
    train_model_name,
    run_time=5,
)
