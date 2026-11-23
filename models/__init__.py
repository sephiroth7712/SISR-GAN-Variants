from models.ESPCN import ESPCN
from models.SRGAN import SRGAN
from models.EDSR import EDSR
from models.FSRCNN import FSRCNN
from models.IDN import IDN

# Generator classes, keyed by the name passed to --model_name.
GENERATORS = {
    "espcn": ESPCN,
    "fsrcnn": FSRCNN,
    "idn": IDN,
    "srgan": SRGAN,
    "edsr": EDSR,
}


def get_generator(model_name):
    try:
        return GENERATORS[model_name]
    except KeyError:
        raise ValueError(
            "Unknown model name '%s'. Choose one of: %s"
            % (model_name, ", ".join(sorted(GENERATORS)))
        ) from None
