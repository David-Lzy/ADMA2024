from CODE.Utils.package import *
from CODE.Utils.utils import *
from CODE.Utils.constant import *
from CODE.Attack.mix import Mix


class PGD(Mix):
    def __init__(self, **kwargs):
        super().__init__(swap=False, CW=False, kl_loss=False, **kwargs)

        self.attack_method_path = "PGD"

        self.out_dir = os.path.join(
            ATTACK_OUTPUT_PATH,
            self.train_method_path,
            self.attack_method_path,
            self.dataset,
        )
