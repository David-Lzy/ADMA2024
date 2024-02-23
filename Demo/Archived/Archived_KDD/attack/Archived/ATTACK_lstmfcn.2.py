import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from __init__ import *
from CODE.Train.lstm_fcn import *
from CODE.Attack.mix import Mix

logging.getLogger().setLevel(logging.INFO)
trainer = Trainer(
    dataset=UNIVARIATE_DATASET_NAMES[0], epoch=100, model=LSTMFCN, unbais=True
)
torch.backends.cudnn.enabled = False

for j in range(1, 4):
    for name, i_parameter_dict in ATTACK_METHODS.items():
        print(name, i_parameter_dict)
        for i, dataset in enumerate(UNIVARIATE_DATASET_NAMES):
            if i % 2 == 0:
                continue
            attacker = Mix(
                dataset=dataset,
                model=LSTMFCN,
                batch_size=64,
                train_method_path=trainer.method_path,
                eps_init=0.01,
                **i_parameter_dict,
                path_parameter=os.path.join(name, f"run_{j}"),
            )
            attacker.perturb_all(
                to_device=True,
                override=True,
            )
            attack_method = os.path.join(
                trainer.method_path, attacker.attack_method_path
            )
        concat_metrics_attack(method=attack_method, datasets=UNIVARIATE_DATASET_NAMES)
