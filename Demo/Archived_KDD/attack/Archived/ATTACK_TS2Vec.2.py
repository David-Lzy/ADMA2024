import sys
import os


def parent_dir(path, n):
    for _ in range(n):
        path = os.path.dirname(path)
    return path


sys.path.append(parent_dir(__file__, 3))
from __init__ import *

from CODE.Attack.mix import Mix
from CODE.Train.TS_2_V.main import *


def ATTACK_ALL(
    train_model_name,
    run_time=3,
    attack_method_dict=ATTACK_METHODS,
    datasets_names=UNIVARIATE_DATASET_NAMES,
    gpu_id="cuda:0",
    your_gpu_number=1,
    this_run_gpu_index=0,
):
    if this_run_gpu_index >= your_gpu_number:
        raise ValueError("this_run_gpu_index >= your_gpu_number")
    for j in range(1, run_time + 1):
        for name, i_parameter_dict in attack_method_dict.items():
            print(j, name, i_parameter_dict)
            for i, dataset in enumerate(datasets_names):
                if i % your_gpu_number == this_run_gpu_index:
                    continue
                attacker = Mix(
                    dataset=dataset,
                    model=train_model_name,
                    batch_size=64,
                    train_method_path=trainer_method_path,
                    eps_init=0.01,
                    **i_parameter_dict,
                    path_parameter=os.path.join(name, f"run_{j}"),
                    device=gpu_id,
                )
                attacker.perturb_all(
                    to_device=True,
                    override=True,
                )
                attack_method = os.path.join(
                    trainer_method_path, attacker.attack_method_path
                )
            concat_metrics_attack(
                method=attack_method, datasets=UNIVARIATE_DATASET_NAMES
            )


train_model_name = Classifier_TS2V
logging.getLogger().setLevel(logging.INFO)
trainer = Trainer(
    model=train_model_name,
)
trainer_method_path = trainer.method_path
del trainer


ATTACK_ALL(
    train_model_name,
    run_time=3,
    gpu_id="cuda:0",
    your_gpu_number=2,
    this_run_gpu_index=1,
)
