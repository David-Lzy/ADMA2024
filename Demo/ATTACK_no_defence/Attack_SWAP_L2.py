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


from CODE.Attack.swap_l2 import SWAPL2

attacker = SWAPL2
special_paramater = {
    "c": 1e-2,
    "epoch": 100,
}
# while True:
#     try:
#         attack_all(
#             attack_class=attacker,
#             reverse=False,
#             override=False,
#             device="cuda:0",
#             special_paramater=special_paramater,
#         )
#         break
#     except KeyboardInterrupt:
#         break
#     except RuntimeError:
#         torch.backends.cudnn.enabled = False

attack_all(
    attack_class=attacker,
    reverse=False,
    override=False,
    device="cuda:0",
    special_paramater=special_paramater,
)
# code3
