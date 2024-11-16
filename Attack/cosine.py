from CODE.Utils.package import *
from CODE.Utils.utils import *
from CODE.Utils.constant import *
from CODE.Attack.attacker import Attack
from CODE.Attack.mix import Mix


# cosine_similarity_coefficient
class csc:
    def __init__(
        self,
    ) -> None:
        super().__init__()

    @staticmethod
    # compute_cosine_similarity
    def cosine_s(input_array1, input_array2):
        """
        Computes the cosine similarity between two input arrays.

        Parameters:
        input_array1 (Tensor): The first input tensor.
        input_array2 (Tensor): The second input tensor, to be compared against the first.

        Returns:
        Tensor: A tensor containing the cosine similarity between the input tensors.
        """
        # 计算向量 A 和 B 的点积
        dot_product = torch.sum(input_array1 * input_array2, dim=-1)

        # 计算向量 A 和 B 的欧几里得范数
        norm_a = torch.norm(input_array1, p=2, dim=-1)
        norm_b = torch.norm(input_array2, p=2, dim=-1)

        # 计算余弦相似度
        cos_similarity = dot_product / ((norm_a * norm_b) + 1e-8)

        return cos_similarity


class COS(Mix):
    default_c1 = 1e-4
    default_c2 = 1e-4

    def __init__(self, **kwargs):
        # Extract 'c_i' from kwargs if it exists
        self.c1 = kwargs.pop("c1", self.default_c1)
        self.c2 = kwargs.pop("c2", self.default_c2)

        # Call the parent class's constructor with the remaining kwargs
        super().__init__(swap=True, swap_index=1, CW=True, kl_loss=False, **kwargs)
        # self.wcf_instance.weights_f()

        self.attack_method_path = "COS"

        self.out_dir = os.path.join(
            ATTACK_OUTPUT_PATH,
            self.train_method_path,
            self.attack_method_path,
            self.dataset,
        )

    def __cos_sim__(self, x, r):
        sim = -torch.log((csc.cosine_s(x, x + r) + 1) / 2)
        # sim = -1 / (1 - csc.cosine_s(x, x + r) + 1e-2)
        return sim

    # Rest of the Correlation class methods...
    def __CW_loss_fun__(self, x, r, y_target, top1_index):
        # Compute weighted FFT integration of r
        cos_los = self.__cos_sim__(x, r)

        # Compute the loss
        y_pred_adv = self.f(x + r)
        loss = self.__LOSS__(y_pred_adv, y_target)

        # Apply mask to exclude incorrect classifications
        mask = torch.zeros_like(loss, dtype=torch.bool)
        _, top1_index_adv = torch.max(y_pred_adv, dim=1)
        for i in range(len(y_target)):
            if not top1_index_adv[i] == top1_index[i]:
                mask[i] = True
        loss[mask] = 0
        # cos_los[mask] = 0

        # Combine the FFT integrated r with the attack loss
        total_loss = cos_los.mean() * self.c1 + loss.mean()

        return total_loss

    def __NoCW_loss_fun__(self, x, r, y_target, top1_index):
        _ = super().__NoCW_loss_fun__(x, r, y_target, top1_index)

        return _ + self.c1 * self.__cos_sim__(x, r).mean()
