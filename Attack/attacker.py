from CODE.Utils.package import *
from CODE.Utils.utils import *
from CODE.Utils.constant import *


class Attack(torch.nn.Module):
    def __init__(
        self,
        dataset=None,
        model=None,
        batch_size=None,
        epoch=None,
        eps_init=None,
        eps=None,
        device=None,
        train_method_path=None,  # know train_method pth location
        path_parameter=None,  # know attack output location
        adeversarial_training=None,
        model_P=None,
        **kwargs,
    ):
        init_params = copy.deepcopy(locals())
        init_params.pop("self")
        self.init_params = init_params
        super().__init__()

        # 使用默认配置更新未提供（即为None）的参数
        self.config = {
            k: v if v is not None else DEFAULT_ATTACK_PARAMETER.get(k)
            for k, v in init_params.items()
        }

        # Override self.config with provided parameters

        for k, v in self.config.items():
            if not k in PRIVATE_VARIABLE:
                setattr(self, k, v)

        if self.adeversarial_training:
            logging.warning(f"adeversarial_training: {adeversarial_training}")

        self.device = (
            self.config["device"]
            if self.config["device"] != None
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        self.model_weight_path = os.path.join(
            TRAIN_OUTPUT_PATH, self.train_method_path, self.dataset, MODEL_NAME
        )  # if not train_method_path is None else train_method_path

        # self.attack_method_path = get_method_loc(
        #     {k: init_params[k] for k in self.path_parameter}
        # )
        # self.out_dir = os.path.join(
        #     ATTACK_OUTPUT_PATH,
        #     self.train_method_path,
        #     self.attack_method_path,
        #     self.dataset,
        # )  # We calso need train_method_path to know who we are attck.

        try:
            self.model_info = torch.load(
                self.model_weight_path, map_location=self.device
            )

            epoch = self.model_info["epoch"]

            train_config = self.model_info["config"]
            self.defence = train_config["defence"]

            try:
                if not train_config["epoch"] == epoch:
                    logging.warning(
                        f"Epoch is not equal to the model_info['epoch'] in {self.model_weight_path}"
                    )
                if not self.train_method_path == train_config["method_path"]:
                    logging.info(
                        f"train_method_path is not equal to the model_info['method_path'] in {self.model_weight_path}"
                    )
            except KeyError:
                pass
        except FileNotFoundError:
            # 兼容旧版
            logging.warning(
                f"Can't find {self.model_weight_path}, try to find {MODEL_NAME} use old method."
            )
            self.model_weight_path = os.path.join(
                TRAIN_OUTPUT_PATH,
                self.train_method_path,
                self.dataset,
                "Done",
                "final_model_weights.pth",
            )
            self.model_info = dict()
            self.model_info["model_state_dict"] = torch.load(
                self.model_weight_path, map_location=self.device
            )
            self.defence = None

        _phase = "TRAIN" if self.adeversarial_training else "TEST"
        self.loader, self.shape, self.nb_classes = load_data(
            self.dataset, phase=_phase, batch_size=self.batch_size
        )[:3]

        init_model(self)

        # Be careful of the order. load_state_dict must after model = Model()
        try:
            self.model.load_state_dict(self.model_info["model_state_dict"])
        except RuntimeError:
            # 假设 self.model_info["model_state_dict"] 是你要加载的权重字典
            state_dict = self.model_info["model_state_dict"]

            # 处理state_dict中的键，移除不需要的前缀
            new_state_dict = {
                key.replace("mother_model.", ""): value
                for key, value in state_dict.items()
            }

            # 使用处理后的state_dict加载权重
            self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.model_name = os.path.basename(__file__).split(".")[0]

    def f(self, x):
        # If the `run` method is not defined, it dynamically assigns the `forward` method of the parent class to `run`.
        # This ensures that the model remains functional even if `run` is not explicitly defined,
        # thereby offering flexibility and a safeguard against method resolution issues in the class inheritance.
        x.to(self.device)
        try:
            return self.model.run(x)
        except AttributeError:
            self.model.run = self.model.forward
            return self.model(x)

    def _loss_function(self, x, r, y_target):
        raise NotImplementedError

    def _get_y_target(self, x):
        raise NotImplementedError

    def __init_r__(self, x):
        raise NotImplementedError

    def __get_optimizer__(self, r):
        return optim.Adam([r], lr=0.0001, betas=(0.9, 0.999), eps=1e-07, amsgrad=True)

    def __perturb__(self, x):
        raise NotImplementedError

    def perturb(self):
        logging.info("_" * 50)
        logging.info(f"Doing: {self.dataset}")
        start = time.time()
        all_perturbed_x = []
        all_perturbed_y = []
        all_predicted_y = []
        self.all_sum_losses = np.zeros(self.epoch)
        self.dist = []

        i = 1
        self.details = dict()
        for batch_id, (x, y) in enumerate(self.loader):
            self.__batch_id__ = batch_id
            self.details[batch_id] = {"x": x.detach().cpu().numpy()}

            logging.debug(f"batch: {i}")
            logging.debug(">" * 50)
            perturbed_x, perturbed_y, predicted_y, sum_losses = self.__perturb__(x)
            perturbed_x = perturbed_x.detach().cpu().numpy()
            perturbed_x = np.squeeze(perturbed_x, axis=1)
            self.dist.extend(
                np.sum((perturbed_x - np.squeeze(x.numpy(), axis=1)) ** 2, axis=1)
            )
            all_perturbed_x.append(perturbed_x)
            perturbed_y = perturbed_y.detach().cpu().numpy()
            all_perturbed_y.append(perturbed_y)
            predicted_y = predicted_y.detach().cpu().numpy()
            all_predicted_y.append(predicted_y)

            self.all_sum_losses += sum_losses
            i += 1

        self.duration = time.time() - start
        self.x_perturb = np.vstack(all_perturbed_x)
        self.y_perturb = np.hstack(all_perturbed_y)
        self.y_predict = np.vstack(all_predicted_y).argmax(axis=1)

    def metrics(self):
        map_ = self.y_perturb != self.y_predict
        self.nb_samples = self.x_perturb.shape[0]

        Count_Success = sum(map_)
        Count_Fail = self.nb_samples - Count_Success
        ASR = Count_Success / self.nb_samples
        # distance = np.hstack(self.dist)
        distance = np.array(self.dist)
        success_distances = distance[map_]
        failure_distances = distance[~map_]

        # Create a dictionary with the data
        self.data = {
            "ASR": ASR,
            "mean_success_distance": np.mean(success_distances),
            "mean_failure_distance": np.mean(failure_distances),
            "overall_mean_distance": np.mean(distance),
            "median_success_distance": np.median(success_distances),
            "median_failure_distance": np.median(failure_distances),
            "overall_median_distance": np.median(distance),
            "Count_Success": Count_Success,
            "Count_Fail": Count_Fail,
            "duration": self.duration,
        }

    def perturb_all(self, override=False, to_device=False):
        _ = folder_contains_files(
            self.out_dir,
            "results.csv",
            "x_perturb.tsv",
            "y_perturb.npy",
            "loss.txt",
        )
        if to_device and (not override) and _:
            logging.info(f"Dataset: {self.dataset} exist, skip!")
            load_data_from_csv(self)
            return
        self.perturb()
        self.metrics()
        if to_device:
            create_directory(self.out_dir)
            save_perturb(self)
            save_conf_to_json(self.out_dir, self.finished_params)

    def plot_comparison(self, index, original_file=None, perturbed_file=None):
        perturbed_file = (
            os.path.join(self.out_dir, "x_perturb.tsv")
            if perturbed_file is None
            else perturbed_file
        )
        original_file = (
            os.path.join(DATASET_PATH, self.dataset, f"{self.dataset}_TEST.tsv")
            if original_file is None
            else original_file
        )

        # 读取数据
        original_data = pd.read_csv(original_file, sep="\t", header=None)
        perturbed_data = pd.read_csv(perturbed_file, sep="\t", header=None)

        # 获取特定索引的样本
        original_sample = original_data.iloc[index][
            1:
        ]  # 假设第一个元素是标签或其他非数据项
        original_sample = original_sample[1:].reset_index(drop=True)
        perturbed_sample = perturbed_data.iloc[index]

        # 绘制图形进行对比
        plt.figure(figsize=(12, 6))
        plt.title(f"Original vs Perturbed Sample (Index: {index})")
        plt.plot(original_sample, label="Original Sample")
        plt.plot(perturbed_sample, label="Perturbed Sample", linestyle="--")
        plt.legend()
        plt.show()
