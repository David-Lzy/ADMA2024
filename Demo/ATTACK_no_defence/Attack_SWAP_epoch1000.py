import sys
import os

path = os.path.abspath(__file__)
while True:
    try:
        sys.path.append(path)
        from __init__ import *
    except ModuleNotFoundError:
        sys.path.remove(path)
        path = os.path.dirname(path)
    else:
        break


from CODE.Attack.swap import SWAP


MODEL_DICT1 = {
    "DEFAULT": Classifier_INCEPTION,
    "LSTMFCN": LSTMFCN,
    "Classifier_MACNN": Classifier_MACNN,
    "ClassifierResNet18": ClassifierResNet18,
}


def attack_all(
    attack_class,
    reverse=False,
    override=False,
    device="cuda:0",
    special_paramater=dict(),
):
    attack_method = ""
    from CODE.Utils.constant import MODEL_DICT
    from CODE.Train.trainer import Trainer

    def sub_attack(batch_size, device="cuda:0"):
        attacker = attack_class(
            dataset=dataset,
            model=i_train_model,
            batch_size=batch_size,
            train_method_path=trainer_method_path,
            eps_init=0.01,
            device=device,
            **special_paramater,
        )
        attacker.perturb_all(
            to_device=True,
            override=override,
        )
        attack_method = os.path.join(trainer_method_path, attacker.attack_method_path)
        return attack_method

    datasets = UNIVARIATE_DATASET_NAMES[::-1] if reverse else UNIVARIATE_DATASET_NAMES

    model_list = unique_and_ordered(MODEL_DICT1.values())
    model_list = model_list[::-1] if reverse else model_list
    for i_train_model in model_list:
        trainer = Trainer(
            model=i_train_model,
        )
        trainer_method_path = trainer.method_path
        del trainer
        for i, dataset in enumerate(datasets):
            torch.cuda.empty_cache()
            dbs = 128
            while True:
                try:
                    attack_method = sub_attack(batch_size=dbs, device=device)
                    break
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        dbs = int(max(dbs - 16, dbs * 0.8))
                        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                            "expandable_segments:True"
                        )
                        logging.warning(
                            f"CUDA out of memory, try smaller batch: {dbs - 16}..."
                        )
                        logging.debug(str(e))
                        torch.cuda.empty_cache()
                        time.sleep(10)

                        if dbs < 1:
                            raise RuntimeError(str(e))
                    elif (
                        "cudnn RNN backward can only be called in training mode"
                        in str(e)
                    ):
                        logging.warning(str(e))
                        torch.backends.cudnn.enabled = False
                    else:
                        raise RuntimeError(str(e))
        concat_metrics_attack(method=attack_method, datasets=UNIVARIATE_DATASET_NAMES)


attacker = SWAP
special_paramater = {
    # "c": 1e-3,
    "epoch": 1000,
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


# code4
