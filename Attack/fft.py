from CODE.Utils.package import *
from CODE.Utils.utils import *
from CODE.Utils.constant import *
from CODE.Attack.attacker import Attack
from CODE.Attack.mix import Mix
from CODE.Attack.correlation import *


# weighted_fft_coefficient
class wfc(wcf):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    # weighted_fft_integration
    def wfi(self, input_array, weights=None):
        """
        Applies FFT to the input array, then weights it and integrates the result.

        Parameters:
        input_array (Tensor): The input tensor to apply FFT.
        weights (Tensor, optional): The weights to apply after FFT. If None, use self.weights.

        Returns:
        Tensor: The integrated result after applying FFT and weights.
        """
        # Apply FFT
        input_fft = torch.fft.fft(input_array).real

        # Apply weights
        if weights is None:
            if self.weights is None:
                weights = self.weights_f(input_array.shape[-1])
            else:
                weights = self.weights
        weighted_fft = input_fft * weights.to(input_array.device)

        # mean the weighted FFT result
        integrated_fft = torch.mean(weighted_fft, dim=-1)

        return integrated_fft


# wcf: 'weighted_fft_coefficient_Gaissian'
class wfc_G(wfc):
    def __init__(
        self,
        # parameter for weight.
        mu=0,
        sigma=1,
    ) -> None:
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.weights = None

    def weights_f(self, length):
        # Generate Gaussian weights
        x = torch.arange(0, length, dtype=torch.float32)
        weights = torch.exp(-0.5 * ((x - self.mu * length) / self.sigma) ** 2)
        weights /= weights.max()  # Normalize to range [0, 1]
        self.weights = weights
        return weights


# wcf: 'weighted_fft_coefficient_Sigmoid'
class wfc_S(wfc):
    def __init__(
        self,
        # parameters for weight using Sigmoid function.
        steepness=1,
        midpoint=0.5,
    ) -> None:
        super().__init__()
        self.steepness = steepness
        self.midpoint = midpoint
        self.weights = None

    def weights_f(self, length):
        # Generate Sigmoid weights_f
        x = torch.arange(0, length, dtype=torch.float32)
        weights = 1 / (1 + torch.exp(-self.steepness * (x - self.midpoint * length)))
        self.weights = weights
        return weights


class FFT(Correlation):
    def __init__(self, **kwargs):
        # Call the parent class's constructor with the remaining kwargs
        super().__init__(swap=True, swap_index=1, kl_loss=False, CW=True, **kwargs)

        self.attack_method_path = "FFT"
        self.out_dir = os.path.join(
            ATTACK_OUTPUT_PATH,
            self.train_method_path,
            self.attack_method_path,
            self.dataset,
        )

    def __CW_loss_fun__(self, x, r, y_target, top1_index):
        # Compute weighted FFT integration of r
        fft_integrated_r = self.wcf_instance.wfi(r)

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
        fft_integrated_r[mask] = 0

        # Combine the FFT integrated r with the attack loss
        total_loss = fft_integrated_r.mean() * self.c1 + loss.mean()

        return total_loss

    def __NoCW_loss_fun__(self, x, r, y_target, top1_index):
        fft_integrated_r = self.wcf_instance.wfi(r)
        y_pred_adv = self.f(x + r)
        loss = self.__LOSS__(y_pred_adv, y_target)
        total_loss = fft_integrated_r.mean() * self.c1 + loss.mean() + self.c2 * torch.norm(r, p=2)
        return total_loss
