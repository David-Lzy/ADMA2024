from CODE.Utils.package import *
from CODE.Utils.utils import *
from CODE.Utils.constant import *
from CODE.Attack.attacker import Attack


class Mix(Attack):
    def __init__(
        self,
        # parameter for Attacker
        dataset=None,
        model=None,
        batch_size=None,
        epoch=None,
        eps=None,
        device=None,
        train_method_path=None,  # know train_method pth location
        path_parameter=None,  # know attack output location
        # parameter for swap
        swap=None,
        swap_index=None,
        gamma=None,
        # parameter for Kullback Leibler
        kl_loss=None,
        # parameter for Attack Gradient Descent
        CW=None,
        c=None,
        # parameter for init r
        eps_init=None,
        # parameter for BIM
        sign_only=None,
        alpha=None,
        adeversarial_training=None,
        model_P=None,
        make_demo=None,
        **kwargs,
    ):
        init_params = copy.deepcopy(locals())
        init_params.pop("self")

        self.config = {
            k: v if v is not None else DEFAULT_ATTACK_PARAMETER.get(k)
            for k, v in init_params.items()
        }
        for k, v in self.config.items():
            if not k in PRIVATE_VARIABLE:
                setattr(self, k, v)

        super().__init__(**init_params)

        self.__init_r__ = self.__init_r__
        # 这一行不能删除，否则会报错

        self.attack_method_path = (
            get_method_loc({k: init_params[k] for k in self.path_parameter})
            if type(self.path_parameter) == list
            else get_method_loc(self.path_parameter)
        )
        self.out_dir = os.path.join(
            ATTACK_OUTPUT_PATH,
            self.train_method_path,
            self.attack_method_path,
            self.dataset,
        )  # We calso need train_method_path to know who we are attck.

        self.finished_params = copy.deepcopy(locals())
        self.finished_params.pop("self")
        ############## init finished ##############

    def _get_y_target(self, *args, **kwargs):
        if self.swap:
            return self.__get_y_target_SWAP__(*args, **kwargs)
        else:
            return self.__get_y_target_RAND__(*args, **kwargs)

    def __loss_function__(self, *args, **kwargs):
        if self.CW:
            return self.__CW_loss_fun__(*args, **kwargs)
        else:
            return self.__NoCW_loss_fun__(*args, **kwargs)

    def __perturb__(self, *args, **kwargs):
        if self.sign_only:
            return self.__perturb_s__(*args, **kwargs)
        else:
            return self.__perturb_g__(*args, **kwargs)

    def __init_r__(self, x):
        r_data = (
            torch.randint(2, x.shape, dtype=x.dtype, device=x.device) * 2 - 1
        ) * self.eps_init
        r = torch.nn.Parameter(r_data, requires_grad=True)
        return r

    def __get_y_target_RAND__(self, inputs):
        with torch.no_grad():
            # Compute predictions for inputs
            predictions = self.f(inputs)
            # Get the indices of the maximum predicted class
            _, predicted_classes = torch.max(predictions, dim=1)
            # Initialize target tensor based on whether KL loss is used
            targets = (
                torch.zeros_like(predictions)
                if not self.kl_loss
                else predictions.clone()
            )

            for i in range(len(predictions)):
                # Get all class indices except for the predicted class
                alternative_classes = torch.arange(
                    predictions.shape[1], device=predictions.device
                )
                alternative_classes = alternative_classes[
                    alternative_classes != predicted_classes[i]
                ]
                # Randomly select a new class from alternatives
                new_class = alternative_classes[
                    torch.randint(0, len(alternative_classes), (1,))
                ]
                # Set target for the selected class to 1.0
                targets[i, new_class] = 1.0

        return targets, predicted_classes

    def __get_y_target_SWAP__(self, inputs):
        with torch.no_grad():
            # Compute predictions for inputs
            predictions = self.f(inputs)
            # Get the indices of the top classes for swapping
            _, top_class_indices = torch.topk(predictions, self.swap_index + 1, dim=1)
            # Initialize target tensor based on whether KL loss is used
            targets = (
                torch.zeros_like(predictions)
                if not self.kl_loss
                else predictions.clone()
            )

            for i in range(len(predictions)):
                # Indices of top classes for current prediction
                top_indices = top_class_indices[i]
                # Compute mean of the highest and swap_index-th predicted values
                mean_value = (
                    predictions[i, top_indices[0]]
                    + predictions[i, top_indices[self.swap_index]]
                ) / 2
                # Adjust target values for swap_index-th and highest class based on mean_value and gamma
                # targets[i, top_indices[self.swap_index]] = mean_value + self.gamma
                # targets[i, top_indices[0]] = mean_value - self.gamma
                targets[i, top_indices[self.swap_index]] = 1
                targets[i, top_indices[0]] = 0

        return targets, top_class_indices[:, 0]

    def __LOSS__(self, y_pred_adv, y_target):
        return nn.functional.cross_entropy(y_pred_adv, y_target, reduction="none")

    def __ce_kl_mixed_LOSS__(self, y_pred_adv, y_target):
        soft_label = self._get_label()

        _ce = self.__cross_entropy_LOSS__(y_pred_adv, y_target)
        _kl = self.__kullback_leibler_LOSS__(y_pred_adv, y_target)
        if self.iepoch == 0:
            self.balance = _ce.mean() / _kl.mean()
        return (
            self.iepoch * _ce + self.balance * _kl.mean(-1) * (self.epoch - self.iepoch)
        ) / self.epoch

    def __CW_loss_fun__(self, x, r, y_target, top1_index):
        y_pred_adv = self.f(x + r)
        loss = self.__LOSS__(y_pred_adv, y_target)

        mask = torch.zeros_like(loss, dtype=torch.bool)
        _, top1_index_adv = torch.max(y_pred_adv, dim=1)

        for i in range(len(y_target)):
            if not top1_index_adv[i] == top1_index[i]:
                mask[i] = True
        loss[mask] = 0

        # Combine the attack loss with the L2 regularization
        l2_reg = torch.norm(r, p=2)

        return l2_reg * self.c + loss.mean()

    def __NoCW_loss_fun__(self, x, r, y_target, top1_index):
        y_pred_adv = self.f(x + r)
        return self.__LOSS__(y_pred_adv, y_target).mean()

    def __perturb_g__(self, x):
        x = x.to(self.device)  # Move x to the device first
        y_pred = self.f(x)
        self.details[self.__batch_id__]["y_pred"] = y_pred.detach().cpu().numpy()
        r = self.__init_r__(x)
        optimizer = self.__get_optimizer__(r)
        y_target, top1_index = self._get_y_target(x)
        # 这里看起来不需要to_device
        sum_losses = np.zeros(self.epoch)

        for epoch in range(self.epoch):
            self.iepoch = epoch
            loss = self.__loss_function__(x, r, y_target, top1_index)
            optimizer.zero_grad()

            loss.backward(retain_graph=True)

            optimizer.step()
            r.data = torch.clamp(r.data, -self.eps, self.eps)
            sum_losses[epoch] += loss.item()
            if not (epoch + 1) % 100:
                logging.debug(f"Epoch: {epoch+1}/{self.epoch}")

            if self.make_demo:
                self.details[self.__batch_id__][epoch] = {
                    "loss": loss.item(),
                    "x_adv": (x + r).cpu().detach().numpy(),
                    "y_adv": self.f(x + r).cpu().detach().numpy(),
                }

        x_adv = x + r
        y_adv = self.f(x_adv).argmax(1)

        return x_adv, y_adv, y_pred, sum_losses

    def __perturb_s__(self, x):
        x = x.to(self.device)  # Move x to the device first
        y_pred = self.f(x)
        r = self.__init_r__(x)
        optimizer = self.__get_optimizer__(r)
        y_target, top1_index = self._get_y_target(x)
        sum_losses = np.zeros(self.epoch)

        for epoch in range(self.epoch):
            self.iepoch = epoch
            loss = self.__loss_function__(x, r, y_target, top1_index)
            optimizer.zero_grad()
            loss.backward()

            # Here, we use the sign of the gradient for the update
            grad_sign = r.grad.sign()
            r.data = r.data - self.alpha * grad_sign
            # alpha is your step size for BIM
            r.data = torch.clamp(r.data, -self.eps, self.eps)

            sum_losses[epoch] += loss.item()
            if not (epoch + 1) % 100:
                logging.debug(f"Epoch: {epoch+1}/{self.epoch}")

        x_adv = x + r
        y_adv = self.f(x_adv).argmax(1)

        return x_adv, y_adv, y_pred, sum_losses

    def __perturb_soft__(self, x):
        def __CW__(x, r, top1_index):
            y_pred_adv = self.f(x + r)

            target_logits = y_pred_adv[torch.arange(y_pred_adv.size(0)), top1_index]
            loss = -torch.log(1 - target_logits.clamp(min=1e-6))

            mask = torch.zeros_like(loss, dtype=torch.bool)
            _, top1_index_adv = torch.max(y_pred_adv, dim=1)

            for i in range(len(y_pred_adv)):
                if not top1_index_adv[i] == top1_index[i]:
                    mask[i] = True
            loss[mask] = 0

            # Combine the attack loss with the L2 regularization
            l2_reg = torch.norm(r, p=2)

            return l2_reg * self.c + loss.mean()

        x = x.to(self.device)  # Move x to the device first
        y_pred = self.f(x)
        r = self.__init_r__(x)
        optimizer = self.__get_optimizer__(r)
        _, top1_index = torch.max(y_pred, dim=1)
        sum_losses = np.zeros(self.epoch)

        for epoch in range(self.epoch):
            loss = __CW__(x, r, top1_index)
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            r.data = torch.clamp(r.data, -self.eps, self.eps)
            sum_losses[epoch] += loss.item()
            if not (epoch + 1) % 100:
                logging.debug(f"Epoch: {epoch+1}/{self.epoch}")

        x_adv = x + r
        y_adv = self.f(x_adv).argmax(1)

        return x_adv, y_adv, y_pred, sum_losses
