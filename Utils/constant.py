# Get the current working directory
from CODE.Utils.package import *


def setup_logging():
    import datetime
    import logging
    import os

    hostname = socket.gethostname()
    username = getpass.getuser()
    current_datetime = datetime.datetime.now()
    formatted_log = current_datetime.strftime("%Yy_%mm_%dd_%Hh_%Mm_%Ss.log")
    formatted_log = f"{hostname}_{username}_" + formatted_log

    log_path = os.path.join(HOME_LOC, "LOG", formatted_log)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path)],
    )


# Set up logging
setup_logging()
logger = logging
# to set a different logging level for a particular logger
# Using: logging.getLogger().setLevel(logging.XXXX)
# XXXX can be: DEBUG, INFO, WARNING, ERROR, CRITICAL


if not os.getcwd() == HOME_LOC:
    logging.warning("Home path not equal to work path, changing!")
    os.chdir(HOME_LOC)
print("HOME_LOC:", HOME_LOC)


# 检查是否有可用的GPU，否则使用CPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义数据和输出路径
DATASET_PATH = os.path.join(HOME_LOC, "DATA", "UCRArchive_2018")
ADVERSARIAL_TRAINING_PATH = os.path.join(HOME_LOC, "DATA", "ADVERSARIAL")
ATTACK_OUTPUT_PATH = os.path.join(HOME_LOC, "OUTPUT", "attack")
TRAIN_OUTPUT_PATH = os.path.join(HOME_LOC, "OUTPUT", "train")

# 定义文件名常量
# DONE_NAME = ""
# DOING_NAME = ""
MODEL_NAME = "MODEL_INFO.pth"

# 加载默认训练参数配置
# 加载默认训练参数配置

DEFAULT_TRAIN_PARAMETER_PATH = os.path.join(
    HOME_LOC, "CODE", "Config", "DEFAULT_TRAIN.json"
)

DEFAULT_TRAIN_PARAMETER = {
    "dataset": "Beef",
    "device": None,
    "batch_size": 64,
    "epoch": 750,
    "loss": "CrossEntropyLoss",
    "override": False,
    "defense": {"angle": False, "get_w_mtx": False, "Augmentation": {}},
    "path_parameter": ["defense"],
    "adversarial_training": False,
    "adversarial_path": "",
    "adversarial_resume": False,
}

DEFAULT_ATTACK_PARAMETER = {
    "dataset": "Beef",
    "device": DEVICE,
    "model": "Classifier_INCEPTION",
    "batch_size": 64,
    "epoch": 1000,
    "eps_init": 0.001,
    "eps": 0.1,
    "gamma": 0.01,
    "c": 0.00001,
    "swap": False,
    "swap_index": 1,
    "kl_loss": False,
    "CW": False,
    "sign_only": False,
    "alpha": 0.01,
    "angle": False,
    "path_parameter": ["swap", "kl_loss", "CW"],
    "adversarial_training": False,
}  # 初始化一个空字典，用于存储配置

UNIVARIATE_DATASET_NAMES = [
    "Beef",
]


ATTACK_METHODS = {
    "BIM": {
        "swap": False,
        "kl_loss": False,
        "CW": False,
        "sign_only": True,
        "alpha": 1e-3,
        "epoch": 1000,
    },
    "FGSM": {
        "swap": False,
        "kl_loss": False,
        "CW": False,
        "sign_only": True,
        "alpha": 1e-1,
        "epoch": 1,
    },
    "GM_PGD": {
        "swap": False,
        "kl_loss": False,
        "CW": False,
        "sign_only": False,
        "epoch": 1000,
    },
    "GM_PGD_L2": {
        "swap": False,
        "kl_loss": False,
        "CW": True,
        "sign_only": False,
        "epoch": 1000,
    },
    "SWAP": {
        "swap": True,
        "kl_loss": True,
        "CW": False,
        "sign_only": False,
        "epoch": 1000,
    },
    "SWAP_L2": {
        "swap": True,
        "kl_loss": True,
        "CW": True,
        "sign_only": False,
        "epoch": 1000,
    },
    "FSWAP": {
        "swap": True,
        "kl_loss": True,
        "CW": False,
        "sign_only": True,
        "alpha": 1e-1,
        "epoch": 1,
    },
    "RSWAP": {
        "swap": False,
        "kl_loss": True,
        "CW": False,
        "sign_only": False,
        "epoch": 1000,
    },
    "RSWAP_L2": {
        "swap": False,
        "kl_loss": True,
        "CW": True,
        "sign_only": False,
        "epoch": 1000,
    },
}

try:
    with open(DEFAULT_TRAIN_PARAMETER_PATH, "r", encoding="utf-8") as file:
        DEFAULT_TRAIN_PARAMETER = json.load(file)
except FileNotFoundError:
    logger.error(f"File not found: {DEFAULT_TRAIN_PARAMETER_PATH}")

# 加载默认攻击参数配置
DEFAULT_ATTACK_PARAMETER_PATH = os.path.join(
    HOME_LOC, "CODE", "Config", "DEFAULT_ATTACK.json"
)
try:
    with open(DEFAULT_ATTACK_PARAMETER_PATH, "r", encoding="utf-8") as file:
        DEFAULT_ATTACK_PARAMETER = json.load(file)
except FileNotFoundError:
    logger.error(f"File not found: {DEFAULT_ATTACK_PARAMETER_PATH}")

# 加载默认数据集名称配置
UNIVARIATE_DATASET_NAMES_PATH = os.path.join(
    HOME_LOC, "CODE", "Config", "DEFAULT_DATA_NAME.json"
)

# 初始化一个空字典，用于存储配置
try:
    with open(UNIVARIATE_DATASET_NAMES_PATH, "r", encoding="utf-8") as file:
        UNIVARIATE_DATASET_NAMES = json.load(file)
except FileNotFoundError:
    logger.critical(f"File not found: {UNIVARIATE_DATASET_NAMES_PATH}")


# 添加了更具描述性的变量名，同时使用常规命名约定
# print(
#     type(DEFAULT_TRAIN_PARAMATER),
#     type(DEFAULT_ATTACK_PARAMATER),
#     type(UNIVARIATE_DATASET_NAMES),)


from CODE.Train.inception_time import Classifier_INCEPTION
from CODE.Train.lstm_fcn import LSTMFCN
from CODE.Train.macnn import Classifier_MACNN
from CODE.Train.resnet import ClassifierResNet18
from CODE.Train.TS_2_V.main import Classifier_TS2V

MODEL_DICT = {
    "Classifier_INCEPTION": Classifier_INCEPTION,
    "Default": Classifier_INCEPTION,
    "default": Classifier_INCEPTION,
    "DEFAULT": Classifier_INCEPTION,
    "LSTMFCN": LSTMFCN,
    "Classifier_MACNN": Classifier_MACNN,
    "ClassifierResNet18": ClassifierResNet18,
    "Classifier_TS2V": Classifier_TS2V,
}

TRAIN_MODEL_LIST = set([*MODEL_DICT.values()])

PRIVATE_VARIABLE = [
    "self",
    "__class__",
    "model",
]


def variable_cleanner(variables):
    init_params = copy.deepcopy(variables)
    for i in PRIVATE_VARIABLE:
        try:
            init_params.pop(i)
        except KeyError:
            pass
    return init_params


if __name__ == "__main__":
    print(
        type(DEFAULT_TRAIN_PARAMETER),
        type(DEFAULT_ATTACK_PARAMETER),
        type(UNIVARIATE_DATASET_NAMES),
    )
    print(
        DEFAULT_TRAIN_PARAMETER,
        DEFAULT_ATTACK_PARAMETER,
        UNIVARIATE_DATASET_NAMES,
    )
