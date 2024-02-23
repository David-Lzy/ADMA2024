import sys
import os


def parent_dir(path, n):
    for _ in range(n):
        path = os.path.dirname(path)
    return path


sys.path.append(parent_dir(__file__, 3))
from __init__ import *

method_path = ""
from CODE.Train.inception_time import *

for i, dataset in enumerate(UNIVARIATE_DATASET_NAMES):
    trainer = Trainer(
        dataset=dataset,
        epoch=1000,
        model=Classifier_INCEPTION,
        unbais=True,
        device="cuda:0",
    )
    print(trainer.method_path)
    method_path = trainer.method_path
    break
    # trainer.train_and_evaluate(to_device=True, override=True)
    # pprint(trainer.train_result)

concat_metrics_train(mode="train", method=method_path)
concat_metrics_train(mode="test", method=method_path)
