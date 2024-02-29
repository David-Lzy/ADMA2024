import sys
import os

pwd = os.path.dirname(__file__)
path = pwd
while True:
    try:
        sys.path.append(path)
        from __init__ import *
    except ModuleNotFoundError:
        sys.path.remove(path)
        path = os.path.dirname(path)
    else:
        break
print("pwd:", pwd)

from CODE.Utils.augmentation import Augmentation
from CODE.Train.defence import Defence

for i, i_dataset in enumerate(UNIVARIATE_DATASET_NAMES):
    # if not i % 3 == 0:
    #     continue
    for i_model in TRAIN_MODEL_LIST:
        defence_model = Defence
        aug_method = "gaussian_smooth"
        defence_model_paramaters = {
            "mother_model": i_model,
            "augmentation": Augmentation.get_method()[aug_method],
            "aug_paramater": dict(),
        }
        i_batch = 256
        while i_batch > 32:
            try:
                trainer = Trainer(
                    dataset=i_dataset,
                    epoch=1000,
                    batch_size=i_batch,
                    model=defence_model,
                    unbais=True,
                    device="cuda:0",
                    model_P=defence_model_paramaters,
                )
                trainer.path_parameter = {"aug_method": aug_method}
                trainer.__set_output_dir__()
                trainer.train_and_evaluate(to_device=True, override=False)
                break
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logging.debug(str(e))
                    torch.cuda.empty_cache()
                    time.sleep(1)
                    i_batch = int(i_batch - 32)
                    if i_batch < 32:
                        raise RuntimeError(str(e))
                elif "cudnn RNN backward can only be called in training mode" in str(e):
                    logging.warning(str(e))
                    torch.backends.cudnn.enabled = False
                else:
                    raise RuntimeError(str(e))

        concat_metrics_train(mode="train", method=trainer.method_path)
        concat_metrics_train(mode="test", method=trainer.method_path)
