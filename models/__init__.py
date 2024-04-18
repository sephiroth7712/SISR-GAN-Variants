from models.common import resolve
from models.common import resolve_single
from models.common import evaluate
from models.ESPCN import ESPCN
from models.SRGAN import SRGAN
from models.EDSR import EDSR
from models.FSRCNN import FSRCNN
from models.IDN import IDN


def get_generator(model_name):
    if model_name == "espcn":
        return ESPCN
    elif model_name == "srgan":
        return SRGAN
    elif model_name == "edsr":
        return EDSR
    elif model_name == "fsrcnn":
        return FSRCNN
    elif model_name == "idn":
        return IDN
    else:
        return ESPCN
