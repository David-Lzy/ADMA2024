import sys
import os


def parent_dir(path, n):
    for _ in range(n):
        path = os.path.dirname(path)
    return path


sys.path.append(parent_dir(__file__, 3))
from __init__ import *
from CODE.Train.TS_2_V.main import *

method_path = ""

for i, dataset in enumerate(UNIVARIATE_DATASET_NAMES[1:]):
    # if i % 2 == 1:
    #     continue
    trainer = Trainer(
        dataset=dataset,
        epoch=100,
        model=Classifier_TS2V,
        unbais=True,
        # device="cuda:0",
        batch_size=256,
    )
    trainer.optimizer = optim.Adam(trainer.model.parameters(), lr=0.005)
    trainer.scheduler = ReduceLROnPlateau(
        trainer.optimizer,
        mode="min",
        factor=1.0 / (2 ** (1 / 3)),
        patience=50,
        min_lr=1e-5,
        cooldown=0,
    )

    trainer.train_and_evaluate(to_device=True, override=True)
    method_path = trainer.method_path


concat_metrics_train(mode="train", method=method_path)
concat_metrics_train(mode="test", method=method_path)
