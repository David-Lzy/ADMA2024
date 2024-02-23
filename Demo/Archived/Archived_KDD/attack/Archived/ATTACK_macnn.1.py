import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from __init__ import *
from CODE.Train.inception_time import *
from CODE.Attack.mix import Mix


from CODE.Train.macnn import *

train_model_name = Classifier_MACNN

logging.getLogger().setLevel(logging.INFO)
trainer = Trainer(
    dataset=UNIVARIATE_DATASET_NAMES[0],
    epoch=100,
    model=train_model_name,
    unbais=True,
)
trainer_method_path = trainer.method_path
del trainer
# torch.backends.cudnn.enabled = False


for j in range(1, 4):
    for name, i_parameter_dict in ATTACK_METHODS.items():
        print(j, name, i_parameter_dict)
        for i, dataset in enumerate(UNIVARIATE_DATASET_NAMES):
            if i % 2 == 1:
                continue
            attacker = Mix(
                dataset=dataset,
                model=train_model_name,
                batch_size=64,
                train_method_path=trainer_method_path,
                eps_init=0.01,
                **i_parameter_dict,
                path_parameter=os.path.join(name, f"run_{j}"),
                device="cuda:0",
            )
            attacker.perturb_all(
                to_device=True,
                override=True,
            )
            attack_method = os.path.join(
                trainer_method_path, attacker.attack_method_path
            )
        concat_metrics_attack(method=attack_method, datasets=UNIVARIATE_DATASET_NAMES)
