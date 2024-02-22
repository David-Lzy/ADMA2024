from CODE.Utils.package import *
from CODE.Utils.utils import *
from CODE.Utils.constant import *
from CODE.Attack.mix import Mix


class SWAPL2(Mix):
    def __init__(self, **kwargs):
        super().__init__(swap=True, swap_index=1, CW=False, kl_loss=False, **kwargs)

        self.attack_method_path = "SWAPL2"

        self.out_dir = os.path.join(
            ATTACK_OUTPUT_PATH,
            self.train_method_path,
            self.attack_method_path,
            self.dataset,
        )

    def __loss_function__(self, x, r, y_target, top1_index):
        y_pred_adv = self.f(x + r)
        loss = self.__LOSS__(y_pred_adv, y_target).mean()
        l2_reg = torch.norm(r, p=2)
        return l2_reg * self.c + loss.mean()
