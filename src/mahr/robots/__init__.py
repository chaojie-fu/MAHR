"""Built-in Python robot configuration classes."""
from .g1 import UnitreeG1Config
from .h1 import UnitreeH1Config
from .gr1 import FourierGR1Config
from .n1 import FourierN1Config
from .x1 import AgibotX1Config
from .berkeley import BerkeleyHumanoidConfig
from .atlas import AtlasHDConfig
from .icub import ICubConfig

PRESETS = {
    'g1': UnitreeG1Config,
    'h1': UnitreeH1Config,
    'gr1': FourierGR1Config,
    'n1': FourierN1Config,
    'x1': AgibotX1Config,
    'berkeley': BerkeleyHumanoidConfig,
    'atlas': AtlasHDConfig,
    'icub': ICubConfig,
}
