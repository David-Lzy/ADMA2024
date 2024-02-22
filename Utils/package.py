import os
import sys

try:
    import re
    import csv
    import math
    import copy
    import json
    import time
    import shutil
    import logging
    import inspect
    import tempfile
    import socket
    import getpass

    import numpy as np
    import pandas as pd

    from pprint import pprint
    from datetime import datetime
    from sklearn.metrics import precision_score, recall_score, f1_score
    from sklearn.utils.class_weight import compute_class_weight

    import torch
    import torch.optim as optim
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import CrossEntropyLoss
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import TensorDataset, DataLoader

    import matplotlib.pyplot as plt
    from collections.abc import Iterable
    from collections import OrderedDict

    HOME_LOC = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sys.path.append(HOME_LOC)

except ModuleNotFoundError as e:
    print(e)
    sys.exit(-1)
