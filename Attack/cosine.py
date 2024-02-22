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
        cos_similarity = dot_product / (norm_a * norm_b)

        return cos_similarity


class COS(Mix):
    default_c1 = 1e-4
    default_c2 = 1e-4

    def __init__(self, **kwargs):
        # Extract 'c_i' from kwargs if it exists
        self.c1 = kwargs.pop("c1", self.default_c1)
        self.c2 = kwargs.pop("c2", self.default_c2)

        # Call the parent class's constructor with the remaining kwargs
        super().__init__(swap=True, swap_index=1, CW=False, kl_loss=False, **kwargs)
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
        return sim

    # Rest of the Correlation class methods...
    def __CW_loss_fun__(self, x, r, y_target, top1_index):
        _ = super().__CW_loss_fun__(x, r, y_target, top1_index)
        return self.c1 * self.__cos_sim__(x, r).mean() + _

        # # Combine the attack loss with the corr regularization
        # # corr = self.wcf_instance.wcc(x, x + r)
        # l2_reg = torch.norm(r, p=2)
        # # print(corr_diff.mean().item(), loss.mean().item(), "\n")
        # # print(r.shape)
        # # print(corr.shape, loss.shape, loss.mean().shape, l2_reg.shape)
        # return l2_reg * self.c2 + loss.mean()

    def __NoCW_loss_fun__(self, x, r, y_target, top1_index):
        _ = super().__NoCW_loss_fun__(x, r, y_target, top1_index)

        return _ + self.c1 * self.__cos_sim__(x, r).mean()
