from CODE.Utils.package import *


class Augmentation:
    Avoid_name = ["get_method", "get_index"]

    def __init__(self) -> None:
        self.methods = {
            name: method
            for name, method in inspect.getmembers(self, predicate=inspect.isfunction)
            # if isinstance(self.__dict__.get(name, None), staticmethod)
        }

    @staticmethod
    def Jitter(x, p=0.9, amplitude=0.2):
        sign_array = (torch.randint(0, 2, size=x.shape) * 2 - 1).to(x.device)
        binary_array = (torch.rand(x.shape) < p).float().to(x.device)
        spike_array = sign_array * binary_array * amplitude
        x_jitter = x + spike_array

        return x_jitter.to(x.device)

    @staticmethod
    def JitterWithDecay(x, p=0.5, amplitude=0.2):
        # 生成符号数组 (-1 或 1)
        sign_array = (torch.randint(0, 2, size=x.shape) * 2 - 1).to(x.device)

        # 生成二进制数组，决定哪些位置将添加扰动
        binary_array = (torch.rand(x.shape) < p).float().to(x.device)

        # 生成扰动值
        spike_array = sign_array * binary_array * amplitude

        # 创建一个衰减函数，使得扰动随着时间逐渐减小
        # 假设 x 的最后一个维度是时间维度
        decay_factor = torch.linspace(1, 0, x.shape[-1]).to(x.device)
        decay_factor = decay_factor.expand_as(x)

        # 将衰减因子应用到扰动值上
        decayed_spike_array = spike_array * decay_factor

        # 添加扰动到原始数据
        x_jittered = x + decayed_spike_array

        return x_jittered.to(x.device)

    @staticmethod
    def binomial_mask(x, keep_prob=0.75):
        mask = torch.from_numpy(np.random.binomial(1, keep_prob, size=x.shape)).to(
            torch.bool
        )
        masked_x = x * mask.float().to(x.device)

        return masked_x

    @staticmethod
    def continuous_mask(
        x,
        max_chunk_ratio=0.05,
        overall_mask_ratio=0.25,
    ):
        length = x.shape[-1]
        max_mask_length = int(max_chunk_ratio * length)
        total_mask_length = int(overall_mask_ratio * length)

        masked_arr = x.clone()
        current_mask_length = 0

        while current_mask_length < total_mask_length:
            start = torch.randint(0, length, (1,)).item()
            mask_length = torch.randint(1, max_mask_length + 1, (1,)).item()

            if start + mask_length > length:
                mask_length = length - start
            if current_mask_length + mask_length > total_mask_length:
                mask_length = total_mask_length - current_mask_length
            if len(x.shape) == 3:
                masked_arr[:, :, start : start + mask_length] = 0
            else:
                masked_arr[:, start : start + mask_length] = 0

            current_mask_length += mask_length

        return masked_arr

    @staticmethod
    def gaussian_noise(x, mean=0, std=0.1):
        size = x.shape
        mean_tensor = torch.full(size, mean).float().to(x.device)
        noise_array = torch.normal(mean_tensor, std).to(x.device)
        noised_x = x + noise_array

        return noised_x

    @staticmethod
    def gaussian_smooth(x, kernel_size=10, sigma=5):
        # Make sure x has the shape [batch_size, 1, sample_length]
        assert (
            len(x.shape) == 3 and x.shape[1] == 1
        ), "Expected input shape: [batch_size, 1, sample_length]"

        # Create a Gaussian kernel
        gauss_kernel = torch.exp(
            -torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size) ** 2
            / (2 * sigma**2)
        )
        gauss_kernel = gauss_kernel / gauss_kernel.sum()  # Normalize

        # Adapt to the input dimension
        gauss_kernel = gauss_kernel.view(1, 1, -1).to(x.device)

        padding_size = kernel_size // 2
        # Calculate the mean of the first and last padding_size elements
        start_mean = x[:, :, :padding_size].mean(dim=-1, keepdim=True)
        end_mean = x[:, :, -padding_size:].mean(dim=-1, keepdim=True)

        # Create padding tensors
        start_padding = start_mean.expand(-1, 1, padding_size)
        end_padding = end_mean.expand(-1, 1, padding_size)

        # Concatenate padding tensors with the input tensor x
        padded_x = torch.cat([start_padding, x, end_padding], dim=-1)

        # Perform convolution with padding set to 0, as we have manually added padding
        smoothed_x = F.conv1d(padded_x, gauss_kernel, padding=0)

        # Adjust the length of smoothed_x
        smoothed_x = 0.5 * (smoothed_x[:, :, :-1] + smoothed_x[:, :, 1:])

        return smoothed_x

    @staticmethod
    def correlation(x):
        # Ensure the input is float for FFT
        x = x.float()

        # Normalize x (Mean Subtraction)
        x_normalized = (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True)

        # Perform FFT
        x_fft = torch.fft.fft(x)

        # Compute power spectrum (autocorrelation in frequency domain)
        power_spectrum = x_fft * torch.conj(x_fft)

        # Compute inverse FFT to get the autocorrelation in time domain
        autocorr = torch.fft.ifft(power_spectrum).real

        # Since autocorrelation is symmetric, we can take the first half
        autocorr = autocorr[:, :, : x.shape[-1]]

        autocorr = (autocorr - autocorr.mean(dim=-1, keepdim=True)) / autocorr.std(
            dim=-1, keepdim=True
        )

        return autocorr

    @staticmethod
    def correlation2(x, percentage_range=(10, 100)):
        # Ensure the input is float for FFT
        x = x.float()

        # Normalize x (Mean Subtraction)
        x_normalized = (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True)

        # Perform FFT
        x_fft = torch.fft.fft(x)

        # Compute power spectrum (autocorrelation in frequency domain)
        power_spectrum = x_fft * torch.conj(x_fft)

        # Compute inverse FFT to get the autocorrelation in time domain
        autocorr = torch.fft.ifft(power_spectrum).real

        # Calculate the indices to slice the autocorrelation tensor based on the specified percentage range
        start_idx = int(x.shape[-1] * percentage_range[0] / 100)
        end_idx = int(x.shape[-1] * percentage_range[1] / 100)

        # Slice the autocorrelation tensor
        autocorr = autocorr[:, :, start_idx:end_idx]

        # Since autocorrelation is symmetric, we can take the first half
        autocorr = autocorr[:, :, : x.shape[-1]]

        autocorr = (autocorr - autocorr.mean(dim=-1, keepdim=True)) / autocorr.std(
            dim=-1, keepdim=True
        )

        return autocorr

    @staticmethod
    def correlation3( x, percentage_range=(80, 100), jitter_params={"p": 0.9, "amplitude": 0.4}
    ):
        # Ensure the input is float for FFT
        x = x.float()

        # Normalize x (Mean Subtraction)
        x_normalized = (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True)

        # Perform FFT
        x_fft = torch.fft.fft(x_normalized)

        # Compute power spectrum (autocorrelation in frequency domain)
        power_spectrum = x_fft * torch.conj(x_fft)

        # Compute inverse FFT to get the autocorrelation in time domain
        autocorr = torch.fft.ifft(power_spectrum).real

        # Calculate the indices to slice the autocorrelation tensor based on the specified percentage range
        start_idx = int(x.shape[-1] * percentage_range[0] / 100)
        end_idx = int(x.shape[-1] * percentage_range[1] / 100)

        # Apply jitter to parts of the data outside the specified percentage range
        if start_idx > 0:
            autocorr[:, :, :start_idx] = Augmentation.JitterWithDecay(
                autocorr[:, :, :start_idx], **jitter_params
            )
            autocorr[:, :, :start_idx] = Augmentation.JitterWithDecay(
                autocorr[:, :, :start_idx], **jitter_params
            )
        if end_idx < x.shape[-1]:
            autocorr[:, :, end_idx:] = Augmentation.JitterWithDecay(
                autocorr[:, :, end_idx:], **jitter_params
            )
            autocorr[:, :, end_idx:] = Augmentation.JitterWithDecay(
                autocorr[:, :, end_idx:], **jitter_params
            )

        # Normalize the autocorrelation
        autocorr = (autocorr - autocorr.mean(dim=-1, keepdim=True)) / autocorr.std(
            dim=-1, keepdim=True
        )

        return autocorr
    
    @staticmethod
    def correlation4(
        x, 
        percentage_range=(
            80, 100),
        jitter_params={
            "p": 0.9, 
            "amplitude": 0.4}
    ):
        # Ensure the input is float for FFT
        x = x.float()

        # Normalize x (Mean Subtraction)
        x_normalized = (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True)

        # Perform FFT
        x_fft = torch.fft.fft(x_normalized)

        # Compute power spectrum (autocorrelation in frequency domain)
        power_spectrum = x_fft * torch.conj(x_fft)

        # Calculate the indices to slice the autocorrelation tensor based on the specified percentage range
        start_idx = int(x.shape[-1] * percentage_range[0] / 100)
        end_idx = int(x.shape[-1] * percentage_range[1] / 100)
        # Apply jitter to parts of the data outside the specified percentage range

        if start_idx > 0:
            power_spectrum[:, :, :start_idx] = Augmentation.JitterWithDecay(
                power_spectrum[:, :, :start_idx], **jitter_params
            )
            power_spectrum[:, :, :start_idx] = Augmentation.JitterWithDecay(
                power_spectrum[:, :, :start_idx], **jitter_params
            )
        if end_idx < x.shape[-1]:
            power_spectrum[:, :, end_idx:] = Augmentation.JitterWithDecay(
                power_spectrum[:, :, end_idx:], **jitter_params
            )
            power_spectrum[:, :, end_idx:] = Augmentation.JitterWithDecay(
                power_spectrum[:, :, end_idx:], **jitter_params
            )

        # Compute inverse FFT to get the autocorrelation in time domain
        autocorr = torch.fft.ifft(power_spectrum).real

        # Calculate the indices to slice the autocorrelation tensor based on the specified percentage range
        start_idx = int(x.shape[-1] * percentage_range[0] / 100)
        end_idx = int(x.shape[-1] * percentage_range[1] / 100)



        # Normalize the autocorrelation
        autocorr = (autocorr - autocorr.mean(dim=-1, keepdim=True)) / autocorr.std(
            dim=-1, keepdim=True
        )

        return autocorr
    
    @staticmethod
    def fft(x, percentage_range=(50, 100), jitter_params={"p": 0.9, "amplitude": 0.4}
    ):
        # Ensure the input is float for FFT
        x = x.float()

        # Normalize x (Mean Subtraction)
        x_normalized = (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True)

        # Perform FFT
        x_fft = torch.fft.fft(x_normalized).real

        # Calculate the indices to slice the autocorrelation tensor based on the specified percentage range
        start_idx = int(x.shape[-1] * percentage_range[0] / 100)
        end_idx = int(x.shape[-1] * percentage_range[1] / 100)

        # Calculate the indices to slice the autocorrelation tensor based on the specified percentage range
        start_idx = int(x.shape[-1] * percentage_range[0] / 100)
        end_idx = int(x.shape[-1] * percentage_range[1] / 100)

        # Apply jitter to parts of the data outside the specified percentage range
        if start_idx > 0:
            x_fft[:, :, :start_idx] = Augmentation.JitterWithDecay(
                x_fft[:, :, :start_idx], **jitter_params
            )
            x_fft[:, :, :start_idx] = Augmentation.JitterWithDecay(
                x_fft[:, :, :start_idx], **jitter_params
            )
        if end_idx < x.shape[-1]:
            x_fft[:, :, end_idx:] = Augmentation.JitterWithDecay(
                x_fft[:, :, end_idx:], **jitter_params
            )
            x_fft[:, :, end_idx:] = Augmentation.JitterWithDecay(
                x_fft[:, :, end_idx:], **jitter_params
            )
            
        x_fft = (x_fft - x_fft.mean(dim=-1, keepdim=True)) / x_fft.std(
            dim=-1, keepdim=True
        )

        return x_fft


    @staticmethod
    def nothing(x):
        return x

    @staticmethod
    def get_method(model=None):
        if model == None:
            model = Augmentation
        return {
            name: method
            for name, method in inspect.getmembers(model, predicate=inspect.isfunction)
            if isinstance(model.__dict__.get(name, None), staticmethod)
            and (not name in Augmentation.Avoid_name)
        }

    @staticmethod
    def get_index():
        return list(Augmentation.get_method().keys())


if __name__ == "__main__":
    print(Augmentation.get_method())
