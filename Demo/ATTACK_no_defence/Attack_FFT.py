import sys
import os


def parent_dir(path, n):
    for _ in range(n):
        path = os.path.dirname(path)
    return path


for i in range(1, 100):
    try:
        sys.path.append(parent_dir(__file__, i))
        from __init__ import *
    except ModuleNotFoundError:
        pass
    else:
        logging.getLogger().setLevel(logging.INFO)
        break

from CODE.Attack.fft import FFT
attacker = FFT
from CODE.Attack.fft import wfc_S
special_paramater = {
    "c": 1e1,
    "epoch": 100,
    'corr_parameter': {
        "weight_fun": wfc_S,
        "para": {"steepness": 0.1, "midpoint": 0.1},
    },
}

while True:
    try:
        attack_all(
            attack_class=attacker,
            reverse=False,
            override=False,
            device="cuda:0",
            special_paramater=special_paramater,
        )
        break
    except KeyboardInterrupt:
        break
    except RuntimeError:
        torch.backends.cudnn.enabled = False


# code6
