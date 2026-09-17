# -*- coding: utf-8 -*-
"""
Access to the data files in the dc3 code base.

``dc3`` uses the :mod:`astropy.utils.data` caching system to limit the size of
its distribution, downloading reference files on demand.  This module provides
the class used to access those files.  :mod:`~dc3.pkg.cache` implements the
low-level cache interface.

Every time ``dc3`` is imported, a new
:class:`~dc3.pkg.dc3data.DC3DataPaths` instance is created and used to define
paths to data files.  "Data files" here means anything in the ``dc3/data``
directory tree.  For example:

.. code-block:: python

    from dc3 import dataPaths
    f = dataPaths.tests.get_file_path('U06918_merge_c.fits')

.. important::

    Always use :func:`~dc3.pkg.dc3data.DC3DataPath.get_file_path` rather than
    the ``/`` operator to access a data file.  Behind the scenes it looks for
    the file in the package distribution and downloads it to the cache if
    needed; ``/`` only works for files that are already on disk.

.. note::

    Adapted from ``pypeit/pkg/pypeitdata.py`` in `PypeIt
    <https://github.com/pypeit/PypeIt>`__ (BSD 3-Clause); see
    ``licenses/README.rst``.  The ``s3_cloud`` host is removed, since ``dc3``
    hosts no data there.
"""

import pathlib
import shutil
import warnings

# NOTE: To avoid circular imports, do not import anything from dc3 into this
# module!  Only import from modules in this directory (dc3/pkg).
from .exceptions import DC3Error, DC3PathError
from . import cache


class DC3DataPath:
    """
    Interface between a dc3 data directory and the rest of the code, regardless
    of whether the directory's contents are shipped in the package distribution
    or fetched through the cache.

    Parameters
    ----------
    subdirs : str, `pathlib.Path`
        The subdirectory within the main ``dc3/data`` directory that contains
        the data.
    remote_host : str, optional
        The remote host for the data.  By definition, all files in this data
        path must have the *same* host.  Must be None or ``'github'``.  If None,
        all files in this path are expected to be local to *any* dc3
        installation.

    Attributes
    ----------
    host : str
        String representing the remote host.
    subdirs : str
        The subdirectory path within the ``dc3/data`` directory.
    data : `pathlib.Path`
        Path to the top-level data directory on the user's system.
    path : `pathlib.Path`
        Path to the specific data directory.
    """

    def __init__(self, subdirs, remote_host=None):
        if remote_host not in [None, 'github']:
            raise DC3Error(f'Remote host not recognized: {remote_host}')
        self.host = remote_host
        self.subdirs = subdirs
        self.data = self.check_isdir(pathlib.Path(str(cache.__DC3_DATA__)))
        self.path = self.check_isdir(self.data / self.subdirs)

    def glob(self, pattern):
        """
        Search for all contents of :attr:`path` matching the provided string.

        .. important::

            This *only* finds files that are on disk in the correct directory.
            It does *not* return files that are in the cache or that have not
            yet been downloaded.

        Parameters
        ----------
        pattern : str
            Search string with wildcards.

        Returns
        -------
        generator
            Generator providing the contents matching the search string.
        """
        return self.path.glob(pattern)

    def __repr__(self):
        """Provide a string representation of the path.  Mimics pathlib."""
        return f"{self.__class__.__name__}('{str(self.path)}')"

    def __truediv__(self, p):
        """
        Instantiate a new path object pointing to a subdirectory.

        This should only be used for contents that *exist* in the user's
        distribution.  Any file distributed through the cache should use
        :func:`get_file_path` instead.

        Parameters
        ----------
        p : str, `pathlib.Path`
            A subdirectory or file within :attr:`path`.

        Returns
        -------
        DC3DataPath, `pathlib.Path`
            A :class:`DC3DataPath` if ``p`` is a subdirectory, otherwise a
            `pathlib.Path`.

        Raises
        ------
        DC3PathError
            Raised if the requested contents do not exist.
        """
        if (self.path / p).is_dir():
            # Create a new object; inherit the host from the parent directory
            return DC3DataPath(
                str((self.path / p).relative_to(self.data)), remote_host=self.host
            )
        if (self.path / p).is_file():
            return self.path / p
        raise DC3PathError(
            f'{str(self.path / p)} is not a valid dc3 data path or is a file that does not exist.'
        )

    @staticmethod
    def check_isdir(path):
        """
        Check that the hardwired directory exists.

        Parameters
        ----------
        path : `pathlib.Path`
            The path to check.  This *must* be a directory, not a file.

        Returns
        -------
        `pathlib.Path`
            The input path, if it is valid.

        Raises
        ------
        DC3PathError
            Raised if the path does not exist or is not a directory.
        """
        if not path.is_dir():
            raise DC3PathError(f'Unable to find {path}.  Check your installation.')
        return path

    @staticmethod
    def _parse_format(f):
        """
        Parse the file format, ignoring ``.gz`` extensions.

        Parameters
        ----------
        f : `pathlib.Path`
            File path to parse.

        Returns
        -------
        str
            The extension indicating the file format, with any ``.gz``
            stripped.
        """
        _f = f.with_suffix('') if f.suffix == '.gz' else f
        return _f.suffix.replace('.', '').lower()

    @staticmethod
    def _get_file_path_return(f, return_format, format=None):
        """
        Format the return of :func:`get_file_path`.

        Parameters
        ----------
        f : `pathlib.Path`
            The file path to return.
        return_format : bool
            If True, parse and return the file suffix (e.g., ``'fits'``).
        format : str, optional
            If ``return_format`` is True, override the parsed format with this
            string.  Ignored if None or if ``return_format`` is False.

        Returns
        -------
        `pathlib.Path`, tuple
            The file path and, if requested, the file format.
        """
        if return_format:
            _format = DC3DataPath._parse_format(f) if format is None else format
            return f, _format
        return f

    def get_file_path(
        self, data_file, force_update=False, to_pkg=None, return_format=False, return_none=False
    ):
        """
        Return the path to a file.

        The file must either exist locally or be downloadable from :attr:`host`.
        To *define* a path to a file that meets neither criterion, use
        ``self.path / data_file``.

        Throughout the code base, this is the function that should be used to
        obtain paths to files within :attr:`path`.

        Parameters
        ----------
        data_file : str, `pathlib.Path`
            File name or path.  Must be a file, not a subdirectory within
            :attr:`path`.
        force_update : bool, optional
            If the file is in the cache, force
            `astropy.utils.data.download_file` to update the cache by
            downloading the latest version.
        to_pkg : str, optional
            If the file is in the cache, this affects how the cached file is
            connected to the package installation.  If ``'symlink'``, a symbolic
            link is created in the package directory tree pointing to the cached
            file.  If ``'move'``, the cached file is *moved* (not copied) from
            the cache into the package directory tree.  Anything else, including
            None, performs no operation.  Ignored if ``data_file`` is a valid
            path or a file within :attr:`path`.
        return_format : bool, optional
            If True, return a :obj:`tuple` with the file path and its format
            (e.g., ``'fits'``).
        return_none : bool, optional
            If True, return None if the file does not exist.  If False, an error
            is raised.

        Returns
        -------
        `pathlib.Path`, tuple, None
            The file path and, if requested, the file format.
        """
        # Make sure the file is a Path object
        _data_file = pathlib.Path(data_file).absolute()

        # Check if the file exists on disk, as provided
        if _data_file.is_file():
            # If so, assume this points directly to the file to be read
            return self._get_file_path_return(_data_file, return_format)

        # Otherwise, construct the file name given the root path
        _data_file = self.path / data_file

        # If the file exists, return it
        if _data_file.is_file():
            return self._get_file_path_return(_data_file, return_format)

        # Get the path to the cached file.
        # NOTE: fetch_remote_file only downloads if the file is absent from the
        # cache or force_update is True.
        subdir = str(self.path.relative_to(self.data))
        _cached_file = cache.fetch_remote_file(
            data_file, subdir, force_update=force_update, return_none=return_none
        )
        if _cached_file is None:
            warnings.warn(f'File {data_file} not found in the cache.')
            return None

        # If we've made it this far, the file is being pulled from the cache.
        if to_pkg is None:
            # The cached file is not symlinked or moved, meaning its name on
            # disk is always ``contents``.  So if the format is requested, it
            # must be parsed from the *expected* file name.
            format = DC3DataPath._parse_format(_data_file) if return_format else None
            return self._get_file_path_return(_cached_file, return_format, format=format)

        # Create a symlink to the cached file or move it into the package data
        # directory
        if to_pkg == 'symlink':
            _data_file.symlink_to(_cached_file)
        elif to_pkg == 'move':
            shutil.move(_cached_file, _data_file)
            # ... and delete it from the cache
            cache.remove_from_cache(pattern=data_file)
        return self._get_file_path_return(_data_file, return_format)


class DC3DataPaths:
    """
    The hardwired set of dc3 data paths.

    The top-level directory for all attributes is ``dc3/data``.  Each of these
    directories should, at minimum, contain a README that is version-controlled
    and hosted on GitHub: the code assumes the paths exist, and a
    version-controlled README guarantees that, even when the directory is
    otherwise empty.

    .. important::

        :attr:`defined_paths` must stay in sync with the exclusion rules in
        ``MANIFEST.in``.  This is asserted by a test, not merely documented.
    """

    defined_paths = {
        # Class attribute name
        'tests':
            # Subdirectory in dc3/data
            {'path': 'tests',
             # String name for the remote host; None means the data should be
             # present in *all* installations.
             'host': 'github'},
        # Stellar template libraries
        'templates': {'path': 'templates', 'host': 'github'},
    }
    """
    Dictionary providing the metadata for all the paths defined by the class.
    """

    def __init__(self):
        for key, a in DC3DataPaths.defined_paths.items():
            setattr(self, key, DC3DataPath(a['path'], remote_host=a['host']))

    @classmethod
    def github_paths(cls):
        """
        Return the subset of paths hosted on GitHub.

        Returns
        -------
        dict
            A dictionary with the same format as :attr:`defined_paths`, limited
            to those paths hosted on GitHub.
        """
        return {
            name: meta for name, meta in cls.defined_paths.items() if meta['host'] == 'github'
        }

    @classmethod
    def remote_paths(cls):
        """
        Return the subset of paths with data hosted remotely.

        Returns
        -------
        dict
            A dictionary with the same format as :attr:`defined_paths`, limited
            to those paths with remotely hosted data.
        """
        return cls.github_paths()
