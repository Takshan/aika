from .__version__ import version as __version__

from .core import *
from .utilities import *
from .models import *
from .visual import *

import pkgutil

# import data
# import models
# import utilities
# import visual

__all__ = ['core', 'models', 'utilities', 'visual']


for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    __all__.append(module_name)
    _module = loader.find_module(module_name).load_module(module_name)
    globals()[module_name] = _module