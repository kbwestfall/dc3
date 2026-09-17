"""
dc3 package initialization.

Provides the package-level globals imported by the submodules.
"""

from .pkg.version import version

# Set version
__version__ = version

# Start the log
import logging
from .pkg.logger import get_logger  # noqa: E402
log = get_logger(level=logging.INFO)

# Import and instantiate the data path parser
# NOTE: This *MUST* come after log and __version__ are defined above, hence the
# E402 exemptions; the import order here is load-bearing, not incidental.
from .pkg.dc3data import DC3DataPaths  # noqa: E402
dataPaths = DC3DataPaths()

# Import all the exceptions so that they can be directly imported (e.g., `from
# dc3 import DC3Error`) in all package imports.  The names exported are those in
# dc3.pkg.exceptions.__all__.
from .pkg.exceptions import *  # noqa: E402, F401, F403
