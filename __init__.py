__version__ = "3.0.0"


###########################################################
# Warning, Do not change the order of following packages. #
###########################################################

# try:
from Utils.package import *
from CODE.Utils.constant import *
from CODE.Utils.utils import *

# except ModuleNotFoundError as e:
#     logging.ERROR(e)
#     sys.exit(-1)
