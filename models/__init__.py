from models.common import resolve
from models.common import resolve_single
from models.common import evaluate
from models.ESPCN import ESPCN
from models.SRGAN import SRGAN

def get_generator(model_name):
    if model_name == "espcn":
        return ESPCN
    elif model_name == "srgan":
        return SRGAN
    else:
        return ESPCN