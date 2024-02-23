import sys
import os


def parent_dir(path, n):
    for _ in range(n):
        path = os.path.dirname(path)
    return path


sys.path.append(parent_dir(__file__, 3))
from __init__ import *


from CODE.Train.resnet import *

model_name = ClassifierResNet18
method_path = ""

for i, i_dataset in enumerate(UNIVARIATE_DATASET_NAMES):
    trainer = Trainer(
        dataset=i_dataset,
        epoch=1000,
        batch_size=256,
        model=model_name,
        unbais=True,
    )
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-4)
    trainer.train_and_evaluate(to_device=True, override=False)
    method_path = trainer.method_path

concat_metrics_train(mode="train", method=method_path)
concat_metrics_train(mode="test", method=method_path)
