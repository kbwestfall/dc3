"""
dc3 package initialization.

Provides the package-level globals imported by the submodules.
"""

from .pkg.version import version

# Set version
__version__ = version

# Start the log
import logging
from .pkg.logger import get_logger
log = get_logger(level=logging.INFO)

# Import and instantiate the data path parser
# NOTE: This *MUST* come after log and __version__ are defined above
from .pkg.dc3data import DC3DataPaths
dataPaths = DC3DataPaths()

# Import all the exceptions so that they can be directly imported (e.g., `from
# dc3 import DC3Error`) in all package imports.
from .pkg.exceptions import *
