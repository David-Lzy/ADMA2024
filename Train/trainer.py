from CODE.Utils.package import *
from CODE.Utils.utils import *
from CODE.Utils.constant import *


class Trainer:
    def __init__(
        self,
        dataset=None,
        data_normalize=None,
        device=None,
        batch_size=None,
        epoch=None,
        loss=None,
        unbais=None,
        override=None,
        defence=None,
        path_pramater=None,
        continue_train=False,
        angle=None,
        Augment=None,
        model=None,
        model_P=None,
        adeversarial_training=None,
        adeversarial_path=None,
        adeversarial_resume=None,
        **kwargs,
    ):
        defence = build_defence_dict(
            defence,
            angle,
            Augment,
            adeversarial_training,
            **kwargs,
        )

        init_params = locals()
        init_params.pop("self")

        # Override self.config with provided parameters
        self.config = {
            k: v if v is not None else DEFAULT_TRAIN_PARAMETER.get(k)
            for k, v in init_params.items()
        }

        for k, v in self.config.items():
            if not k in PRIVATE_VARIABLE:
                setattr(self, k, v)

        self.device = (
            self.config["device"]
            if not self.config["device"] is None
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        self.path_parameter = (
            {key: value for key, value in self.config.items() if key in path_pramater}
            if not path_pramater is None
            else {key: self.config[key] for key in self.config["path_pramater"]}
        )  # 增强鲁棒性防止傻逼

        _ = (
            os.path.join(ADVERSARIAL_TRAINING_PATH, self.adeversarial_path)
            if self.adeversarial_training
            else DATASET_PATH
        )
        (
            self.train_loader,
            self.test_loader,
            self.shape,
            _,
            self.nb_classes,
            self.class_weights,
        ) = data_loader(
            self.dataset,
            batch_size=self.batch_size,
            data_path=_,
            normalize=self.data_normalize,
        )

        init_model(self)
        self.__set_output_dir__()

        if loss in [None, "", "default", "Classifier_INCEPTION"]:
            self.loss_function = self.__CE_loss__
            self.__f__ = self.__get_f__
        elif loss in ["angle"]:
            # Prototype_target
            self.w_target = torch.full(
                (self.nb_classes, self.nb_classes), -1 / (self.nb_classes - 1)
            )
            torch.Tensor.fill_diagonal_(self.w_target, 1)
            self.w_target = self.w_target.to(self.device)

            self.loss_function = self.__angle_loss__
            self.__f__ = self.__get_f_no_w__
        else:
            raise KeyError("No right loss chosen!")

        self.__CE__ = (
            CrossEntropyLoss(weight=self.class_weights.to(self.device))
            if unbais
            else CrossEntropyLoss()
        )

        self.optimizer = Adam(self.model.parameters())

        # modified
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=50,
            verbose=True,
            min_lr=0.0001,
        )

        self.resume_dict = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
        }
        self.continue_train = continue_train

        logging.info(f"This Run all parameters: {self.config}")
        logging.debug(f"This Run all parameters: {vars(self)}")

        self.model_info = {
            "architecture": str(self.model),
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "adeversarial_training": self.adeversarial_training,
            "out_dir": self.out_dir,
        }

        ####################### __init_finisehd__ #######################

    def __set_output_dir__(self):
        self.method_path = "model=" + self.model.__class__.__name__ + "_"
        self.method_path += get_method_loc(self.path_parameter)
        self.method_path = (
            self.method_path.replace(
                "adeversarial_training=True",
                f"adeversarial_training_from.{self.adeversarial_path}",
            )
            if self.adeversarial_training
            else self.method_path
        )
        self.out_dir = os.path.join(
            TRAIN_OUTPUT_PATH,
            self.method_path,
            self.dataset,
        )

    def __get_f_no_w__(self, model, x_batch):
        predictions, _ = model(x_batch)
        return predictions

    def __get_f__(self, model, x_batch):
        return model(x_batch)

    def __check_resume__(self, to_device):
        def check_check_point(path):
            checkpoint = torch.load(path, map_location=self.device)
            for k in self.model_info.keys():
                self.resume_dict[k] = checkpoint.get(k)

            if self.resume_dict["architecture"] != str(self.model):
                logging.warning(
                    """Model structure is not match, unable to resume! Please check the model name or set override=True or change the out_dir."""
                )
                return -1, checkpoint

            wanted_e = self.resume_dict["config"]["epoch"]
            real_end_e = self.resume_dict["epoch"]
            this_time_e = self.epoch
            start, end = (
                determine_epochs(wanted_e, real_end_e, this_time_e, self.continue_train)
                if to_device is True
                else (1, wanted_e)
            )
            self.epoch = end
            self.config["epoch"] = end
            return start, checkpoint

        def __resume__(checkpoint: dict):
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logging.info(f"Pth file load from {checkpoint['out_dir']}")

        start = 1
        target_file = os.path.join(self.out_dir, MODEL_NAME)
        ADE_TRAIN_file = os.path.join(
            ADVERSARIAL_TRAINING_PATH, self.adeversarial_path, self.dataset, MODEL_NAME
        )

        if os.path.exists(target_file):
            res = os.path.join(self.out_dir, "test_metrics.csv")
            if self.override:
                logging.info(f"Del task {self.dataset} all files.")
                shutil.rmtree(self.out_dir)
            else:
                start, checkpoint = check_check_point(target_file)
                if not os.path.exists(res) and start == -1:
                    logging.warning(f"File {res} not found, del {target_file}")
                    logging.warning("Broken file Project! Must be override!")
                    self.override = True
                    return self.__check_resume__(to_device)
                if self.adeversarial_training:
                    # This means you are try to resume from a model not used for adeversarial training.
                    if not checkpoint["adeversarial_training"]:
                        logging.warning("ADEVERSARIAL_TRAINING is not match!")
                        os.remove(target_file)
                        logging.info(f"Del task {target_file}.")
                        logging.info(f"Resume from {ADE_TRAIN_file}")
                        if self.adeversarial_resume and os.path.exists(ADE_TRAIN_file):
                            start, checkpoint = check_check_point(ADE_TRAIN_file)
                        else:
                            return start
                __resume__(checkpoint)
        create_directory(self.out_dir)
        return start

    def train_and_evaluate(self, override=False, to_device=True):
        self.override = override
        start_epoch = self.__check_resume__(to_device)
        if start_epoch == -1:
            return
        test_loss_file = open(os.path.join(self.out_dir, "test_loss.txt"), "a")
        logging.info(f"Start locking File {test_loss_file.name}")

        learning_rate_file = open(os.path.join(self.out_dir, "learningRate.txt"), "a")
        logging.info(f"Start locking File {learning_rate_file.name}")

        # current_time = datetime.now()
        # logging.info(f"Current time: {current_time} \n")
        self.start_time = time.time()
        last_saved_time = self.start_time
        for epoch in range(start_epoch, self.epoch + 1):
            self.__train_one_epoch__()
            last_saved_time = self.__save_check_point__(
                epoch,
                last_saved_time,
                test_loss_file,
                learning_rate_file,
                to_device,
            )

            # Evaluation Phase
            test_loss = self.__cal_loss__()

            # Record test loss and learning rate
            test_loss_file.write(f"{test_loss}\n")
            learning_rate_file.write(f"{self.optimizer.param_groups[0]['lr']}\n")
            self.scheduler.step(test_loss)

        test_loss_file.close()
        learning_rate_file.close()

    def __save_check_point__(
        self,
        epoch,
        last_saved_time,
        test_loss_file,
        learning_rate_file,
        to_device,
    ):
        # Save model weights every 50 epochs and delete the old one
        if (time.time() - last_saved_time > 120) or epoch >= self.epoch:
            checkpoint_path = os.path.join(self.out_dir, MODEL_NAME)
            torch.save(
                self.model_info,
                checkpoint_path,
            )
            learning_rate_file.flush()
            test_loss_file.flush()
            last_saved_time = time.time()
        if epoch == self.epoch and to_device:
            self.train_result["duration"] = time.time() - self.start_time
            self.evaluate()
            save_metrics(self.out_dir, "train", self.train_result)
            save_metrics(self.out_dir, "test", self.test_result)
            save_conf_to_json(self.out_dir, vars(self))
            logging.info(f"Task {self.dataset} Finished")
            logging.info("-" * 80)

        return last_saved_time

    def __cal_loss__(
        self,
    ):
        test_loss = 0
        for x_batch, y_batch in self.test_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            loss, _ = self.loss_function(x_batch, y_batch)
            test_loss += loss.item()
        test_loss /= len(self.test_loader)
        return test_loss

    def evaluate(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        test_preds, test_targets = [], []
        with torch.no_grad():
            for x_batch, y_batch in self.test_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                # predictions = self.__f__(model, x_batch)
                loss, predictions = self.loss_function(x_batch, y_batch)
                test_loss += loss.item()
                # test_loss += loss_function(x_batch, y_batch).item()
                pred = predictions.argmax(dim=1, keepdim=True)
                correct += pred.eq(y_batch.view_as(pred)).sum().item()
                test_preds.extend(pred.squeeze().cpu().numpy())
                test_targets.extend(y_batch.cpu().numpy())

            test_loss /= len(self.test_loader)

        accuracy = correct / len(self.test_loader.dataset)
        precision, recall, f1 = metrics(test_targets, test_preds)

        self.test_result = {
            "loss": test_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def __angle_loss__(self, x_batch, y_batch):
        predictions, w_mtx = self.model(x_batch)
        # Prototype Loss
        loss_W = ((w_mtx - self.w_target) ** 2).mean()
        # CE loss
        loss_CE = CrossEntropyLoss()(predictions, y_batch)
        loss_total = loss_W + loss_CE
        return loss_total, predictions

    def __CE_loss__(self, x_batch, y_batch):
        try:
            predictions = self.model.run(x_batch)
        except AttributeError:
            self.model.run = self.model.forward
            predictions = self.model.run(x_batch)
        # CE loss
        loss_CE = self.__CE__(predictions, y_batch)
        return loss_CE, predictions

    def __train_one_epoch__(self):
        self.model.train()

        train_loss = 0
        train_preds, train_targets = [], []
        for x_batch, y_batch in self.train_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            loss, predictions = self.loss_function(x_batch, y_batch)

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_preds.extend(predictions.argmax(dim=1).cpu().numpy())
            train_targets.extend(y_batch.cpu().numpy())

        train_loss /= len(self.train_loader)

        accuracy = np.mean(np.array(train_preds) == np.array(train_targets))
        precision, recall, f1 = metrics(train_targets, train_preds)

        self.train_result = {
            "loss": train_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
