from CODE.Utils.package import *
from CODE.Utils.utils import *
from CODE.Utils.constant import *
from CODE.Attack.attacker import Attack
from CODE.Attack.mix import Mix


class DeepFool(Mix):
    def __init__(self, **kwargs):
        kwargs.setdefault("overshoot", 0.1)
        kwargs["eps_init"] = 0
        init_params = copy.deepcopy(locals())
        init_params.pop("self")
        self.config = {
            k: v if v is not None else DEFAULT_ATTACK_PARAMETER.get(k)
            for k, v in init_params.items()
        }
        for k, v in self.config.items():
            if not k in PRIVATE_VARIABLE:
                setattr(self, k, v)

        super().__init__(**kwargs)

    # def calculate_jacob(self, x, r, y_pred_adv):
    #     jacobians = torch.zeros(self.nb_classes, *x.shape).to(self.device)
    #     for i in range(self.nb_classes):
    #         class_adv_output = y_pred_adv[:, i]
    #         x_adv = x + r
    #         grads = torch.autograd.grad(
    #             class_adv_output,
    #             x_adv,
    #             grad_outputs=torch.ones_like(class_adv_output),
    #             create_graph=True,
    #         )[0]
    #         jacobians[i] = grads
    #     jacobians = jacobians.transpose(0, 1)
    #     return jacobians

    # def select_label(self, jacobians, pre_max):
    #     sample_indices = torch.arange(0, jacobians.shape[0], device=jacobians.device)
    #     selected_ = jacobians[sample_indices, pre_max]
    #     return selected_

    # def __jacob_loss__(self, x, r, y_pred, pre_max, attack_success):
    #     y_pred_adv = self.f(x + r)
    #     _, pre_adv_max = torch.max(y_pred_adv, dim=-1)

    #     attack_success |= pre_adv_max != pre_max
    #     if attack_success.all():
    #         return
    #     ws = self.calculate_jacob(x, r, y_pred_adv)
    #     ws = ws * (~attack_success).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    #     # select_label中参数究竟是什么
    #     f_0 = self.select_label(y_pred_adv, pre_max)
    #     w_0 = self.select_label(jacobians, pre_max)

    #     f = y_pred_adv - f_0.unsqueeze(1)
    #     w = ws - w_0.unsqueeze(1)
    #     # 36行
    #     value = torch.abs(f) / torch.norm(w, p=2, dim=-1).squeeze(-1)
    #     value = torch.where(
    #         torch.isnan(value), torch.tensor(float("inf"), device=value.device), value
    #     )

    #     _, hat_L = torch.min(value, -1)

    #     delta = (
    #         torch.abs(self.select_label(f, hat_L).reshape(-1, 1, 1))
    #         * self.select_label(w, hat_L)
    #         / (torch.norm(self.select_label(w, hat_L), p=2) ** 2)
    #     )

    #     return delta

    # def __perturb__(self, x):
    #     print(1)
    #     print(x.shape)
    #     x = x.to(self.device)  # Move x to the device first
    #     y_pred = self.f(x)
    #     pre_max = y_pred.argmax(-1)
    #     # self.details[self.__batch_id__]["y_pred"] = y_pred.detach().cpu().numpy()
    #     r = self.__init_r__(x)
    #     # optimizer = self.__get_optimizer__(r)
    #     x.requires_grad = True
    #     r.requires_grad = True
    #     sum_losses = np.zeros(self.epoch)
    #     attack_success = torch.zeros(x.shape[0], dtype=torch.bool, device=self.device)

    #     for epoch in range(self.epoch):
    #         loss = self.__jacob_loss__(x, r, y_pred, pre_max, attack_success)
    #         # optimizer.zero_grad()
    #         # loss.backward()
    #         r = torch.where(
    #             attack_success.unsqueeze(1).unsqueeze(2),
    #             r,
    #             r + (1 + self.overshoot) * loss,
    #         )
    #         r.data = torch.clamp(r.data, -self.eps, self.eps)

    #         sum_losses[epoch] += loss.item()
    #         if not (epoch + 1) % 100:
    #             logging.debug(f"Epoch: {epoch+1}/{self.epoch}")

    #     x_adv = x + r
    #     y_adv = self.f(x_adv).argmax(1)
    #     return x_adv, y_adv, y_pred, sum_losses

    # def perturb(self):
    #     logging.info("_" * 50)
    #     logging.info(f"Doing: {self.dataset}")
    #     start = time.time()

    #     all_perturbed_x = []
    #     all_perturbed_y = []
    #     all_predicted_y = []
    #     self.all_sum_losses = np.zeros(self.epoch)
    #     # self.dist = []

    #     for batch_id, (x, y) in enumerate(self.loader):
    #         # self.__batch_id__ = batch_id
    #         # self.details[batch_id] = {
    #         #     "x": x.detach().cpu().numpy()
    #         # }

    #         # logging.debug(f"batch: {i}")
    #         # logging.debug(">" * 50)
    #         perturbed_x, perturbed_y, predicted_y, sum_losses = self.__perturb__(x)
    #         perturbed_x = perturbed_x.detach().cpu().numpy()
    #         perturbed_x = np.squeeze(perturbed_x, axis=1)
    #         all_perturbed_x.append(perturbed_x)
    #         perturbed_y = perturbed_y.detach().cpu().numpy()
    #         all_perturbed_y.append(perturbed_y)
    #         predicted_y = predicted_y.detach().cpu().numpy()
    #         all_predicted_y.append(predicted_y)
    #         self.all_sum_losses += sum_losses

    def perturb(self):
        self.dist = dict

        def calculate_jacob(
            model,
            x,
        ):
            outputs = model(x)
            jacobians = torch.zeros(self.nb_classes, *x.shape).to(self.device)
            for i in range(self.nb_classes):
                class_output = outputs[:, i]
                grads = torch.autograd.grad(
                    class_output,
                    x,
                    grad_outputs=torch.ones_like(class_output),
                    create_graph=True,
                )[0]
                jacobians[i] = grads
            jacobians = jacobians.transpose(0, 1)
            return jacobians

        def select_label(tensor_, labels):
            sample_indices = torch.arange(0, tensor_.shape[0], device=tensor_.device)

            selected_ = tensor_[sample_indices, labels]
            return selected_

        self.dist = []
        start = time.time()
        all_perturbed_x = []
        all_perturbed_y = []
        all_predicted_y = []
        self.all_sum_losses = np.zeros(self.epoch)

        for x_test_tensor, y in self.loader:

            samples = x_test_tensor.to(self.device)
            predicted_y = self.model(samples)
            labels = predicted_y.argmax(-1)
            batch_size = len(samples)

            r = torch.zeros_like(samples)
            overshoot = 0.01
            samples.requires_grad = True
            # r_ = torch.zeros_like(samples)
            attack_success = torch.zeros(
                batch_size, dtype=torch.bool, device=self.device
            )
            for i in range(self.epoch):
                fs = self.model(samples + r)

                _, pre = torch.max(fs, dim=-1)
                attack_success |= pre != labels
                if attack_success.all():
                    break

                # criteria = (pre != labels).reshape(-1,1,1)
                # r_ = r*criteria*(~r_.bool()) + r_

                ws = calculate_jacob(self.model, samples + r)
                ws = ws * (~attack_success).unsqueeze(1).unsqueeze(2).unsqueeze(3)

                f_0 = select_label(fs, labels)
                w_0 = select_label(ws, labels)

                f = fs - f_0.unsqueeze(1)
                w = ws - w_0.unsqueeze(1)

                value = torch.abs(f) / torch.norm(w, p=2, dim=-1).squeeze(-1)
                value = torch.where(
                    torch.isnan(value),
                    torch.tensor(float("inf"), device=value.device),
                    value,
                )

                _, hat_L = torch.min(value, -1)

                delta = (
                    torch.abs(select_label(f, hat_L).reshape(-1, 1, 1))
                    * select_label(w, hat_L)
                    / (torch.norm(select_label(w, hat_L), p=2) ** 2)
                )
                r = torch.where(
                    attack_success.unsqueeze(1).unsqueeze(2),
                    r,
                    r + (1 + overshoot) * delta,
                )

                r = torch.clamp(r, min=-0.1, max=0.1).detach()

            x_adv = (samples + r).detach().cpu().numpy()
            perturbed_x = np.squeeze(x_adv, axis=1)
            y_adv = pre.detach().cpu().numpy()
            self.dist.extend(
                np.sum((x_adv - np.squeeze(x_test_tensor.numpy(), axis=1)) ** 2, axis=1)
            )
            all_perturbed_x.append(perturbed_x)
            all_perturbed_y.append(y_adv)
            predicted_y = predicted_y.detach().cpu().numpy()
            all_predicted_y.append(predicted_y)

        self.duration = time.time() - start
        self.x_perturb = np.vstack(all_perturbed_x)
        self.y_perturb = np.hstack(all_perturbed_y)
        self.y_predict = np.vstack(all_predicted_y).argmax(axis=1)
