from models.common import resolve
from models.common import resolve_single
from models.common import evaluate
from models import ESPCN

def get_generator(model_name):
    if model_name == "espcn":
        return ESPCN
    else:
        return ESPCN