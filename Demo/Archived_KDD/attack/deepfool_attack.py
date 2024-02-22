import sys
import os

def parent_dir(path, n):
    for _ in range(n):
        path = os.path.dirname(path)
    return path
sys.path.append(parent_dir(__file__, 3))
from __init__ import *
from CODE.Attack.deepfool import DeepFool

for i_model in TRAIN_MODEL_LIST:
    if i_model == LSTMFCN:
        torch.backends.cudnn.enabled = False
    else:
        torch.backends.cudnn.enabled = True
    for i_data_set in UNIVARIATE_DATASET_NAMES:
        trainer = Trainer(
            dataset=i_data_set,
            model=i_model,
            unbais=True,
        )
        trainer_method_path = trainer.method_path
        del trainer
        torch.cuda.empty_cache()
        batch_size = 64
        while True:
            try:
                attacker = DeepFool(
                    dataset=i_data_set,
                    model=i_model,
                    train_method_path=trainer_method_path,
                    eps_init=0.0,
                    device="cuda:0",
                    path_parameter="deep_fool",
                    batch_size=batch_size,
                    epoch=5
                )
                attacker.perturb_all(
                    to_device=True,
                    override=True,
                )
                attack_method = os.path.join(trainer_method_path, attacker.attack_method_path)
                break
            except RuntimeError as e:
                del attacker
                torch.cuda.empty_cache()
                print(e)
                batch_size = batch_size - 8
                if batch_size < 8:
                    raise RuntimeError(e)