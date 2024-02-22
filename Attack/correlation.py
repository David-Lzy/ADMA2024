from CODE.Utils.package import *
from CODE.Utils.utils import *
from CODE.Utils.constant import *
from CODE.Attack.attacker import Attack
from CODE.Attack.mix import Mix


# wcf: 'weighted_correlation_coefficient'
class wcf:
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.weights = None

    def normalize(self, data):
        # Ensure data is a float tensor
        data = data.float()
        # Find the min and max values along the last dimension
        min_val = data.min(dim=-1, keepdim=True)[0]
        max_val = data.max(dim=-1, keepdim=True)[0]
        # Perform min-max normalization
        normalized_data = (data - min_val) / (max_val - min_val)
        # Scale to the range [-1, 1]
        normalized_data = normalized_data * 2 - 1
        return normalized_data

    def cross_correlation(self, input_array1, input_array2):
        # Cross-correlation in frequency domain
        input_array1_fft = torch.fft.fft(self.normalize(input_array1))
        input_array2_fft = torch.fft.fft(self.normalize(input_array2))
        # input_array1_fft = torch.fft.fft(input_array1)
        # input_array2_fft = torch.fft.fft(input_array2)
        cross_corr_complex = torch.fft.ifft(
            input_array1_fft * torch.conj(input_array2_fft)
        )
        cross_corr = cross_corr_complex.real
        return cross_corr

    # weighted_correlation_coefficient
    def wcc(self, input_array1, input_array2, weights=None):
        # Cross-correlation in frequency domain
        cross_corr = self.cross_correlation(input_array1, input_array2)

        # Apply Gaussian weights
        weights = (
            self.weights_f(input_array1.shape[-1])
            if self.weights is None
            else self.weights
            if weights is None
            else weights
        ).to(input_array1.device)
        weighted_cross_corr = cross_corr * weights

        # Integrate the weighted cross-correlation
        integrated_cross_corr = torch.sum(weighted_cross_corr, dim=-1)
        return integrated_cross_corr

    def weights_f(self, length):
        raise NotImplementedError

    def visualize(self, log_scale=True, length=1024):
        """
        Visualizes the weights as a plot. If weights are not initialized,
        attempts to generate weights using weights_f with specified length.

        Parameters:
        - log_scale: bool, optional, default True. If True, x-axis will be in log scale.
        - length: int, optional, default 1024. Length to use for weights_f if weights are not initialized.
        """
        if self.weights is None:
            print(
                "Weights not initialized. Attempting to generate weights for visualization."
            )
            print("This is only a demo.")
            # Attempt to generate weights using weights_f with the specified length.
            try:
                self.weights = self.weights_f(length)
            except NotImplementedError:
                print("weights_f method is not implemented.")
                return
            except Exception as e:
                print(f"An error occurred while generating weights: {e}")
                return

        if self.weights is not None:
            # Ensure weights are on CPU and converted to numpy for plotting
            weights_numpy = (
                self.weights.cpu().numpy()
                if self.weights.is_cuda
                else self.weights.numpy()
            )
            plt.plot(weights_numpy)
            plt.title("Weights Visualization")
            plt.xlabel("Index")
            plt.ylabel("Weight Value")
            plt.ylim(-0.05, 1.05)  # 限制 y 轴的范围在 0 到 1
            if log_scale:
                plt.xscale("log")  # 设置 x 轴为对数刻度
            plt.show()


# wcf: 'weighted_correlation_coefficient_Gaissian'
class wcf_G(wcf):
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


# wcf: 'weighted_correlation_coefficient_Sigmoid'
class wcf_S(wcf):
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
        midpoint = self.midpoint if self.midpoint > 1 else self.midpoint * length
        weights = 1 / (1 + torch.exp(-self.steepness * (x - midpoint)))
        self.weights = weights
        return weights


class wcf_S2(wcf):
    def __init__(
        self,
        # parameters for weight using Sigmoid function.
        midpoint=0.5,
    ) -> None:
        super().__init__()
        self.steepness = 0.5 - midpoint
        self.midpoint = midpoint
        self.weights = None

    def weights_f(self, length):
        # Generate Sigmoid weights_f
        x = torch.arange(0, length, dtype=torch.float32)
        midpoint = self.midpoint * length

        weights = self.midpoint / (1 + torch.exp(-self.steepness * (x - midpoint)))
        self.weights = weights

        return weights


class Correlation(Mix):
    default_c1 = 1e-4
    default_c2 = 1e-4
    default_corr_parameter = {
        "weight_fun": wcf_S,
        "para": {"steepness": 1, "midpoint": 0.1},
    }

    def __init__(self, **kwargs):
        # Extract 'corr_parameter' from kwargs if it exists
        corr_parameter = kwargs.pop(
            "corr_parameter", Correlation.default_corr_parameter
        )
        self.c1 = kwargs.pop("c1", self.default_c1)
        self.c2 = kwargs.pop("c2", self.default_c2)
        self.use_l2 = kwargs.pop("use_l2", False)

        # Check if 'corr_parameter' is provided and is a dictionary
        if corr_parameter and isinstance(corr_parameter, dict):
            weight_function = corr_parameter.get(
                "weight_fun", Correlation.default_corr_parameter["weight_fun"]
            )
            weight_params = corr_parameter.get(
                "para", Correlation.default_corr_parameter["para"]
            )

            # Check if the weight function is provided and callable
            if weight_function and callable(weight_function):
                # Initialize the weighting function with the provided parameters
                self.wcf_instance = weight_function(**weight_params)
            else:
                raise ValueError(
                    "Invalid weight function or weight function not provided"
                )
        else:
            self.wcf_instance = None

        # Call the parent class's constructor with the remaining kwargs
        super().__init__(**kwargs)
        # self.wcf_instance.weights_f()

        self.attack_method_path = "CORR"

        self.out_dir = os.path.join(
            ATTACK_OUTPUT_PATH,
            self.train_method_path,
            self.attack_method_path,
            self.dataset,
        )

    # Rest of the Correlation class methods...
    def __CW_loss_fun__(self, x, r, y_target, top1_index):
        corr1 = self.wcf_instance.wcc(x, x + r)
        corr2 = self.wcf_instance.wcc(x, x)
        corr_diff = torch.abs(corr1 - corr2)

        y_pred_adv = self.f(x + r)
        loss = self.__LOSS__(y_pred_adv, y_target)
        # loss = loss * corr_diff * self.c1 / 10

        mask = torch.zeros_like(loss, dtype=torch.bool)
        _, top1_index_adv = torch.max(y_pred_adv, dim=1)

        for i in range(len(y_target)):
            if not top1_index_adv[i] == top1_index[i]:
                mask[i] = True
        loss[mask] = 0

        # Combine the attack loss with the corr regularization
        # corr = self.wcf_instance.wcc(x, x + r)
        l2_reg = torch.norm(r, p=2) if self.use_l2 else 0
        # print(corr_diff.mean().item(), loss.mean().item(), "\n")
        # print(r.shape)
        # print(corr.shape, loss.shape, loss.mean().shape, l2_reg.shape)
        return corr_diff.mean() * self.c1 + l2_reg * self.c2 + loss.mean()

    def __NoCW_loss_fun__(self, x, r, y_target, top1_index):
        corr1 = self.wcf_instance.wcc(x, x + r)
        corr2 = self.wcf_instance.wcc(x, x)
        # corr_diff = torch.abs(corr1 - corr2)

        y_pred_adv = self.f(x + r)
        loss = self.__LOSS__(y_pred_adv, y_target)
        # return corr_diff.mean() * self.c1 + loss.mean()
        # print(corr1.mean().item(), corr2.mean().item(), loss.mean().item())
        return -corr1.mean() * self.c1 + torch.norm(r, p=2) * self.c2 + loss.mean()

    def __perturb_g__(self, x):
        x = x.to(self.device)  # Move x to the device first
        y_pred = self.f(x)
        r = self.__init_r__(x)
        optimizer = self.__get_optimizer__(r)
        y_target, top1_index = self._get_y_target(x)
        sum_losses = np.zeros(self.epoch)

        # avoid calculating the weight every time
        self.wcf_instance.weights_f(x.shape[-1])
        self.wcf_instance.weights.to(self.device)

        for epoch in range(self.epoch):
            loss = self.__loss_function__(x, r, y_target, top1_index)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            r.data = torch.clamp(r.data, -self.eps, self.eps)
            sum_losses[epoch] += loss.item()
            if not (epoch + 1) % 100:
                logging.info(f"Epoch: {epoch+1}/{self.epoch}")

        x_adv = x + r
        y_adv = self.f(x_adv).argmax(1)

        return x_adv, y_adv, y_pred, sum_losses

    def __get_optimizer__(self, r):
        return optim.Adam([r], lr=0.001, betas=(0.9, 0.999), eps=1e-07, amsgrad=True)
