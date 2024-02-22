from CODE.Utils.package import *
from CODE.Utils.utils import *
from CODE.Utils.constant import *
from CODE.Attack.mix import Mix


class SWAP(Mix):
    def __init__(self, **kwargs):
        super().__init__(swap=True, swap_index=1, CW=False, kl_loss=False, **kwargs)

        self.attack_method_path = "SWAP"

        self.out_dir = os.path.join(
            ATTACK_OUTPUT_PATH,
            self.train_method_path,
            self.attack_method_path,
            self.dataset,
        )
