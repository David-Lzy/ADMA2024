from CODE.Utils.package import *
from CODE.Utils.constant import *


def load_data(
    dataset,
    phase="TRAIN",
    batch_size=128,
    data_path=None,
    delimiter="\t",
    normalize=True,
    nfs_mount=True,
):
    def readucr(filename, delimiter=delimiter):
        data = np.loadtxt(filename, delimiter=delimiter)
        Y = data[:, 0]
        X = data[:, 1:]
        return X, Y

    def map_label(y_data):
        unique_classes, inverse_indices = np.unique(y_data, return_inverse=True)
        mapped_labels = np.arange(len(unique_classes))[inverse_indices]
        return mapped_labels

    if data_path == None:
        from CODE.Utils.constant import DATASET_PATH as data_path

    temp_dir = tempfile.gettempdir()
    temp_data_path = os.path.join(temp_dir, dataset)  # 临时文件夹路径
    temp_file_path = os.path.join(temp_data_path, f"{dataset}_{phase}.tsv")
    original_file_path = os.path.join(data_path, dataset, f"{dataset}_{phase}.tsv")
    # 如果临时文件不存在，则从原始数据路径复制
    # 确定文件路径
    if not os.path.exists(temp_file_path) and nfs_mount:
        # 尝试复制到临时目录
        try:
            os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
            shutil.copy(original_file_path, temp_file_path)
        except PermissionError:
            pass  # 如果没有权限复制文件，保持使用原始路径

    # 使用临时文件路径如果存在，否则回退到原始文件路径
    file_path_to_use = (
        temp_file_path if os.path.exists(temp_file_path) else original_file_path
    )

    x, y = readucr(file_path_to_use)

    if normalize:
        x_mean = x.mean(axis=1, keepdims=True)
        x_std = x.std(axis=1, keepdims=True)
        x = (x - x_mean) / (x_std + 1e-8)

    y = map_label(y)
    nb_classes = len(set(y))
    shape = x.shape
    x = x.reshape(shape[0], 1, shape[1])
    x_tensor = torch.tensor(x, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(
        x_tensor, torch.tensor(y, dtype=torch.long)
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(phase == "TRAIN")
    )
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y), y=y
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    return loader, x_tensor.shape, nb_classes, class_weights


def data_loader(dataset, batch_size=128, data_path=None, normalize=True):
    if data_path == None:
        from CODE.Utils.constant import DATASET_PATH as data_path
    (train_loader, train_shape, nb_classes, class_weights) = load_data(
        dataset,
        "TRAIN",
        batch_size=batch_size,
        data_path=data_path,
        normalize=normalize,
    )
    (test_loader, test_shape, nb_classes, _) = load_data(
        dataset,
        "TEST",
        batch_size=batch_size,
        data_path=data_path,
        normalize=normalize,
    )

    return (
        train_loader,
        test_loader,
        train_shape,
        test_shape,
        nb_classes,
        class_weights,
    )


def init_model(self):
    # if model is a string,
    # then it is a model name, else it is a model object
    from CODE.Utils.constant import MODEL_DICT

    self.__abstract_model__ = (
        MODEL_DICT[self.config["model"]]
        if isinstance(self.config["model"], str)
        else self.config["model"]
    )

    if not isinstance(self.model_P, dict):
        raise AttributeError("model_P is not dict!")

    self.model_P["input_shape"] = self.shape
    self.model_P["nb_classes"] = self.nb_classes
    self.model_P["defence"] = self.defence
    self.model_P["device"] = self.device
    self.model_P["dataset"] = self.dataset
    self.model_P["state"] = self.__class__.__name__
    self.model = self.__abstract_model__(**self.model_P).to(self.device)


def metrics(targets, preds):
    precision = precision_score(targets, preds, average="macro", zero_division=0)
    recall = recall_score(targets, preds, average="macro", zero_division=0)
    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    return precision, recall, f1


def create_directory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        logging.info(f"Path {directory_name}' Created")
    else:
        logging.info(f"Path {directory_name}' Existed")


def folder_contains_files(folder_path, *file_names):
    try:
        folder_files = os.listdir(folder_path)
    except FileNotFoundError:
        return False
    file_names_set = set(file_names)
    # 遍历文件名列表，检查是否都存在于文件夹中
    for file_name in file_names_set:
        if file_name not in folder_files:
            return False
    return True


# This function is used to summary all the results of all dataset.
def concat_metrics_train(mode="train", method="", datasets=None):
    if datasets == None:
        datasets = UNIVARIATE_DATASET_NAMES
    metrics_dfs = []
    for dataset in datasets:
        file_path = os.path.join(
            TRAIN_OUTPUT_PATH, method, dataset, f"{mode}_metrics.csv"
        )

        if os.path.exists(file_path):
            dataset_df = pd.DataFrame([dataset], columns=["dataset"])
            temp_df = pd.read_csv(file_path)
            temp_df = pd.concat([dataset_df] + [temp_df], axis=1)

            metrics_dfs.append(temp_df)
        else:
            logging.warning(f"'{file_path}' not found! Skip.")
            return

    final_df = pd.concat(metrics_dfs, ignore_index=False)
    final_df.to_csv(
        os.path.join(
            TRAIN_OUTPUT_PATH,
            f"{mode.upper()}_{'_'.join(method.split(os.path.sep))}_metrics.csv",
        ),
        index=False,
    )


def save_metrics(directory_name, phase, metrics):
    with open(f"{directory_name}/{phase}_metrics.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(metrics.keys())
        writer.writerow(metrics.values())


# This function is used to summary all the attack results of all dataset.
def concat_metrics_attack(method, datasets=None):
    if datasets == None:
        datasets = UNIVARIATE_DATASET_NAMES
    metrics_dfs = []
    for dataset in datasets:
        file_path = os.path.join(ATTACK_OUTPUT_PATH, method, dataset, "results.csv")
        if os.path.exists(file_path):
            dataset_df = pd.DataFrame([dataset], columns=["dataset"])
            temp_df = pd.read_csv(file_path)
            temp_df = pd.concat([dataset_df] + [temp_df], axis=1)

            metrics_dfs.append(temp_df)
        else:
            logging.error(f"{file_path} has no data, ignored!")
            return

    final_df = pd.concat(metrics_dfs, ignore_index=False)
    _ = os.path.join(
        ATTACK_OUTPUT_PATH, f"{'_'.join(method.split(os.path.sep))}_metrics.csv"
    )
    final_df.to_csv(_, index=False)


def check_loop(_):
    try:
        _.__iter__
        if type(_) == str:
            return 0
    except AttributeError:
        return 0
    else:
        return 1


def get_method_loc(methods):
    if type(methods) == dict:
        _new_methos = dict()
        for key, values in methods.items():
            if check_loop(values):
                if len(values) == 0:
                    pass
                else:
                    _new_methos[key] = values
            _new_methos[key] = values
    elif type(methods) in [list, tuple]:
        _new_methos = []
        for values in methods:
            if check_loop(values):
                if len(values) == 0:
                    pass
                else:
                    _new_methos.append(values)
            _new_methos.append(values)
    else:
        _new_methos = methods
    methods = _new_methos
    s_clean = re.sub(r"[{}'\", \[\]]+", "_", str(methods))
    s_clean = re.sub(r"_:_", "=", s_clean)
    s_clean = re.sub(r"__+", "_", s_clean)
    # 移除字符串两边的 "_"
    s_clean = s_clean.strip("_")
    return s_clean


def build_defence_dict(
    defence,
    angle,
    Augment,
    adeversarial_training,
    **kwargs,
):
    def check_sub(_):
        new_Aug = dict()

        if type(_) == dict:
            new_Aug = _
        elif type(_) in (tuple, list):
            if len(_) == 2:
                if [check_loop(_[0]), check_loop(_[1])] == [1, 1]:
                    new_Aug = {
                        "ahead_model": _[0],
                        "in_model": _[1],
                    }
                elif [check_loop(_[0]), check_loop(_[1])] == [0, 0]:
                    new_Aug = {
                        "ahead_model": _,
                    }
            elif not bool(sum([check_loop(i) for i in _])):
                new_Aug = {
                    "ahead_model": _,
                }
            new2_Aug = dict()
            for i, values in new_Aug.items():
                new2_Aug[i] = dict()
                for index, j in enumerate(values):
                    name = Augmentation.get_index()[j]
                    new2_Aug[i][f"{index}.{name}"] = dict()
                    # If you want to change something here,
                    # remember go to Augmentation.py
                    # and  classifier.py __AUG_1__ and __AUG_2__
                    # to change the corresponding code.

            for i in list(new2_Aug.keys()):
                try:
                    if len(new2_Aug[i]) == 0:
                        del new2_Aug[i]
                except TypeError:
                    pass
            new_Aug = new2_Aug
        else:
            raise ValueError("Augmentation is not valid!")
        return new_Aug

    if type(defence) == dict:
        if defence.get("Augmentation", False):
            defence["Augmentation"] = check_sub(defence["Augmentation"])
    else:
        defence = {"Augmentation": dict()}
        for k, v in kwargs.items():
            if k == "Augment":
                defence["Augmentation"] = check_sub(v)
            else:
                defence[k] = v
        if len(defence["Augmentation"]) == 0:
            del defence["Augmentation"]
    # print(defence)
    return defence if len(defence) > 0 else "None"


def determine_epochs(wanted_e, real_end_e, this_time_e, continue_train):
    # 如果实际运行的epoch数量超过了预期的epoch数量
    if real_end_e > wanted_e:
        logging.error(
            f"Epochs not match! {real_end_e} > {wanted_e}. This should never happen!"
        )
        raise ValueError(
            f"Epochs not match! {real_end_e} > {wanted_e}. This should never happen!"
        )

    if continue_train:
        logging.info(
            f"Continuing training from epoch {real_end_e} with an additional {this_time_e} epochs."
        )
        return real_end_e + 1, wanted_e + this_time_e

    # 如果这次的epoch数量小于或等于实际的epoch数量
    if this_time_e < real_end_e:
        _ = f"""You cannot train fewer epochs than the last time! {wanted_e}, {real_end_e} > {this_time_e}. \nIf you want to continue training, please set continue_train=True. \n If you want to retrain, please set overwrite=True."""
        logging.error(_)
        raise ValueError(_)

    # 如果所有的epoch数量都匹配，并且相等
    if real_end_e == wanted_e == this_time_e:
        logging.info(f"Task already done. All epochs match: {real_end_e}.")
        return -1, -1

    # 如果实际的和预期的epoch数量都匹配，但是这次的epoch数量超过了预期的epoch数量
    if real_end_e == wanted_e:
        logging.warning(
            f"Epoch mismatch! Expected {wanted_e}, but got {this_time_e}. Overriding to {this_time_e}."
        )
        return real_end_e + 1, this_time_e

    # 如果实际的epoch数量小于预期的epoch数量
    if real_end_e < wanted_e:
        logging.warning(
            f"The model was not trained to the end in the previous run. Expected {wanted_e}, but got {real_end_e}."
        )
        logging.info(
            f"Resuming training from epoch {real_end_e + 1} for {this_time_e} epochs."
        )
        return real_end_e + 1, this_time_e

    logging.error(f"Unexpected condition encountered.")
    return -1, -1


def save_perturb(self):
    # 保存为CSV文件
    with open(os.path.join(self.out_dir, "results.csv"), mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(self.data.keys())  # 写入列名
        writer.writerow(self.data.values())  # 写入数据

    with open(os.path.join(self.out_dir, "x_perturb.tsv"), "w", newline="") as tsv_file:
        writer = csv.writer(tsv_file, delimiter="\t")
        for row in self.x_perturb:
            writer.writerow(row)

    with open(os.path.join(self.out_dir, "y_perturb.npy"), "wb") as f:
        np.save(f, self.y_perturb)

    # np.save(os.path.join(self.out_dir, "y_perturb.npy"), self.y_perturb)

    with open(os.path.join(self.out_dir, "loss.txt"), "w") as f:
        _ = self.all_sum_losses / self.nb_samples
        all_mean_losses = _.reshape(-1, 1)
        np.savetxt(f, all_mean_losses, delimiter="\t")

    # np.savetxt(
    #     os.path.join(self.out_dir, "loss.txt"), all_mean_losses, delimiter="\t"
    # )
    logging.info(f"Done: {self.dataset}")
    logging.info(">" * 50)


def load_data_from_csv(self):
    """
    从 CSV 文件加载数据，并将其设置为 self.data 的值。
    """
    file_path = os.path.join(self.out_dir, "results.csv")

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        self.data = {}  # 可以设置为一个空字典或其他默认值
        return

    with open(file_path, mode="r", newline="") as file:
        reader = csv.reader(file)
        # 读取列名（第一行）
        keys = next(reader)
        # 读取数据（第二行）
        values = next(reader)
        # 将读取的键值对转换为字典，并设置为 self.data
        self.data = dict(zip(keys, values))
        # 将字符串类型的值转换回适当的类型（假设只有数值和字符串两种类型）
        for key in self.data:
            try:
                # 尝试将数值字符串转换为浮点数
                self.data[key] = float(self.data[key])
            except ValueError:
                # 如果转换失败（即值是非数值字符串），保持原样
                pass


def load_model(self):
    try:
        self.model_info = torch.load(self.model_weight_path)

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
        self.model_info["model_state_dict"] = torch.load(self.model_weight_path)
        self.defence = None

    _phase = "TRAIN" if self.adeversarial_training else "TEST"
    self.loader, self.shape, self.nb_classes = load_data(
        self.dataset, phase=_phase, batch_size=self.batch_size
    )

    self.model = (
        MODEL_DICT[self.config["model"]](
            input_shape=self.shape,
            nb_classes=self.nb_classes,
            defence=self.defence,
        ).to(self.device)
        if isinstance(self.config["model"], str)
        else self.config["model"](
            input_shape=self.shape,
            nb_classes=self.nb_classes,
            defence=self.defence,
        ).to(self.device)
    )
    # if model is a string,
    # then it is a model name, else it is a model object
    # Be careful of the order. load_state_dict must after model = Model()
    self.model.load_state_dict(self.model_info["model_state_dict"])
    self.model.to(self.device)
    self.model.eval()
    self.model_name = os.path.basename(__file__).split(".")[0]


def try_str(value, limit=100):
    try:
        _ = str(value)
        if len(_) < limit:
            return _
        else:
            return _[:limit] + " ..."
    except Exception as e:
        return str(type(value)) + " type, unable to stringify. Error: " + str(e)


def format_timestamp(timestamp):
    """
    将 Unix 时间戳转换为格式化的日期时间字符串。

    :param timestamp: Unix 时间戳，如 1705991301.1124523
    :return: 格式化的日期时间字符串，如 "2024-01-23 16:58:40"
    """
    # 将时间戳转换为 datetime 对象
    dt_object = datetime.fromtimestamp(timestamp)

    # 将 datetime 对象格式化为字符串
    formatted_time = dt_object.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time


def serialize_value(value, i=0):
    """
    递归地检查并序列化值，使其可被JSON序列化。
    """

    if i > 2:
        return try_str(value)
    if isinstance(value, (int, float, str, bool)):
        # 基本数据类型，直接返回
        return value
    elif isinstance(value, dict):
        # 对于字典，递归处理每个值
        return {k: serialize_value(v, i + 1) for k, v in value.items()}
    elif isinstance(value, (list, set, tuple)):
        # 对于其他可迭代类型（如列表、集合、元组），递归处理每个元素
        return [serialize_value(elem, i + 1) for elem in value]
    else:
        # 其他类型，转换为字符串并截断（如果过长）
        return try_str(value)


def save_conf_to_json(filepath, data_dict):
    """
    将字典保存到 JSON 文件中。

    :param filepath: JSON 文件的保存路径。
    :param data_dict: 要保存的字典。
    """
    attrs = {
        "Finished_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    for name, value in data_dict.items():
        if name.startswith("_") or "_loader" in name:
            continue  # 跳过隐藏属性和_loader属性
        value_str = try_str(value, limit=300)
        if len(value_str) > 300 and not name == "config":
            attrs[name] = value_str
        elif "time" in name and isinstance(value, float):
            attrs[name] = format_timestamp(value)
        else:
            attrs[name] = serialize_value(value)

    # 根据value的大小对attrs进行排序
    sorted_attrs = OrderedDict(
        sorted(attrs.items(), key=lambda item: len(str(item[1])))
    )

    # 写入 JSON 文件
    file_name = os.path.join(filepath, "Run_Info.json")
    with open(file_name, "w") as json_file:
        json.dump(sorted_attrs, json_file, indent=4)


def ATTACK_ALL(
    train_model_name,
    run_time=3,
    attack_method_dict=ATTACK_METHODS,
    datasets_names=UNIVARIATE_DATASET_NAMES,
    gpu_id="cuda:0",
    your_gpu_number=1,
    this_run_gpu_index=0,
    override=False,
    reverse=False,
):
    from CODE.Attack.mix import Mix
    from CODE.Train.trainer import Trainer

    logging.getLogger().setLevel(logging.INFO)
    attack_method = ""
    message = ""

    if reverse:
        datasets_names = datasets_names[::-1]
        attack_method_dict = dict(reversed(list(attack_method_dict.items())))

    def sub_attack(batch_size):
        attacker = Mix(
            dataset=dataset,
            model=train_model_name,
            batch_size=batch_size,
            train_method_path=trainer_method_path,
            eps_init=0.01,
            **i_parameter_dict,
            path_parameter=os.path.join(name, f"run_{j}"),
            device=gpu_id,
        )
        attacker.perturb_all(
            to_device=True,
            override=override,
        )
        attack_method = os.path.join(trainer_method_path, attacker.attack_method_path)

    trainer = Trainer(
        model=train_model_name,
    )
    trainer_method_path = trainer.method_path
    del trainer
    torch.cuda.empty_cache()

    if this_run_gpu_index >= your_gpu_number:
        raise ValueError("this_run_gpu_index >= your_gpu_number")
    for j in range(1, run_time + 1):
        for k, (name, i_parameter_dict) in enumerate(attack_method_dict.items()):
            print(j, name, i_parameter_dict)
            for i, dataset in enumerate(datasets_names):
                try:
                    if (i + j + k) % your_gpu_number == this_run_gpu_index:
                        # default_batch_size
                        dbs = 256
                        message = f"GPU {gpu_id}, whit {dataset} {name} {j} {k} {i} "
                        while True:
                            try:
                                sub_attack(batch_size=dbs)
                                break
                            except RuntimeError as e:
                                if "CUDA out of memory" in str(e):
                                    logging.warning(message + str(e))
                                    torch.cuda.empty_cache()
                                    time.sleep(1)
                                    dbs = int(dbs - 32)
                                    if dbs < 32:
                                        raise RuntimeError(message) from e
                except RuntimeError as e:
                    logging.error(message + str(e))
                    torch.cuda.empty_cache()
                    time.sleep(1)
                    continue

            concat_metrics_attack(
                method=attack_method, datasets=UNIVARIATE_DATASET_NAMES
            )


def unique_and_ordered(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def attack_all(
    attack_class,
    reverse=False,
    override=False,
    device="cuda:0",
    special_paramater=dict(),
):
    attack_method = ""
    from CODE.Utils.constant import MODEL_DICT
    from CODE.Train.trainer import Trainer

    def sub_attack(batch_size, device="cuda:0"):
        attacker = attack_class(
            dataset=dataset,
            model=i_train_model,
            batch_size=batch_size,
            train_method_path=trainer_method_path,
            eps_init=0.01,
            device=device,
            **special_paramater,
        )
        attacker.perturb_all(
            to_device=True,
            override=override,
        )
        attack_method = os.path.join(trainer_method_path, attacker.attack_method_path)
        return attack_method

    datasets = UNIVARIATE_DATASET_NAMES[::-1] if reverse else UNIVARIATE_DATASET_NAMES

    model_list = unique_and_ordered(MODEL_DICT.values())
    model_list = model_list[::-1] if reverse else model_list
    for i_train_model in model_list:
        trainer = Trainer(
            model=i_train_model,
        )
        trainer_method_path = trainer.method_path
        del trainer
        torch.cuda.empty_cache()
        for i, dataset in enumerate(datasets):
            dbs = 192
            while True:
                try:
                    attack_method = sub_attack(batch_size=dbs, device=device)
                    break
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        logging.debug(str(e))
                        torch.cuda.empty_cache()
                        time.sleep(1)
                        dbs = int(dbs - 32)
                        if dbs < 32:
                            raise RuntimeError(str(e))
                    elif (
                        "cudnn RNN backward can only be called in training mode"
                        in str(e)
                    ):
                        logging.warning(str(e))
                        torch.backends.cudnn.enabled = False
                    else:
                        raise RuntimeError(str(e))
    concat_metrics_attack(method=attack_method, datasets=UNIVARIATE_DATASET_NAMES)


def log_uniform_random_number(end=100, start=0, num=10):
    # 在对数空间下的均匀分布范围
    log_min = np.log(start + 1)
    log_max = np.log(end)
    linear_uniform_list = []
    for i in range(num):
        # 生成一个在对数空间下均匀分布的随机数
        log_uniform = np.random.uniform(log_min, log_max)
        # 转换回线性空间
        linear_uniform = np.exp(log_uniform)
        linear_uniform_list.append(linear_uniform)
    return linear_uniform_list


if __name__ == "__mian__":
    from CODE.Package import *
    from CODE.Package import HOME_LOC
    from CODE.Utils.constant import *
    from CODE.Utils.constant import UNIVARIATE_DATASET_NAMES as datasets
