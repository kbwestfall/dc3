# -*- coding: utf-8 -*-
"""
Low-level interface to the dc3 file cache.

``dc3`` uses the :mod:`astropy.utils.data` caching system to keep its
distribution small, downloading reference files on demand rather than shipping
them in the wheel.  This module implements the download/cache primitives;
:mod:`~dc3.pkg.dc3data` provides the path registry that the rest of the code
base should actually use.

To find the location of your dc3 cache (by default ``~/.dc3/cache``):

.. code-block:: python

    import astropy.config.paths
    print(astropy.config.paths.get_cache_dir('dc3'))

.. note::

    Adapted from ``pypeit/pkg/cache.py`` in `PypeIt
    <https://github.com/pypeit/PypeIt>`__ (BSD 3-Clause); see
    ``licenses/README.rst``.

    Relative to the PypeIt original this is trimmed in three respects, none of
    which affect how a file is located: ``dc3`` hosts no data on AWS S3, so the
    ``s3_cloud`` host, its hostname lookup and its ``requests`` dependency are
    removed, leaving GitHub as the only remote host; the GitHub repository
    *listing* helpers (which PypeIt uses in its install scripts, via
    ``PyGithub``) are dropped, so ``dc3`` does not depend on ``github``; and the
    most-recent-tag lookup is dropped.  The branch-resolution machinery is
    retained in full -- see :func:`git_branch`.
"""

from functools import reduce
from importlib import resources
import pathlib
import urllib.error
from urllib.parse import urljoin, urlparse
import warnings

import astropy.utils.data

# NOTE: pygit2 is used to identify the checked-out branch; see git_branch for
# why that matters.  It is not a requirement for a general user, hence the try
# block, but it *is* a test requirement (see the `test` extra in
# pyproject.toml).
try:
    from pygit2 import Repository
except ImportError:
    Repository = None
    GitError = None
else:
    from pygit2 import GitError

# NOTE: To avoid circular imports, do not import anything from dc3 into this
# module!  Only import from modules in this directory (dc3/pkg).
from .exceptions import DC3Error
from .version import version as __version__


__DC3_DATA__ = resources.files('dc3') / 'data'
"""Path to the top-level data directory in the package distribution."""

__DC3_REPO_PATH__ = 'kbwestfall/dc3'
"""Default GitHub repository path, used when the local repository is unknown."""

__DC3_DEFAULT_BRANCH__ = 'main'
"""Branch fetched from when the local repository cannot be identified."""


def git_repo():
    """
    Get a reference to the local repository, if possible.

    `pygit2.Repository` performs upward discovery, so this succeeds for any path
    inside the working tree, not only its root.  That is what makes
    :func:`git_branch` work under ``tox``, where the package is installed into a
    virtual environment beneath ``.tox/`` and is therefore still inside the
    checkout.

    Returns
    -------
    `pygit2.Repository`, None
        The repository object, or None if ``pygit2`` is unavailable, ``dc3`` is
        not installed from a git checkout, or the repository has no commits yet.
    """
    if Repository is None:
        # pygit2 not available
        return None
    try:
        return Repository(resources.files('dc3'))
    except GitError:
        # dc3 is not in a git repo, or the repo has no commits
        return None


def git_branch():
    """
    Return the name or hash of the currently checked-out branch.

    .. warning::

        This is **not** cosmetic.  The branch name goes directly into the remote
        URL built by :func:`_build_remote_url`, so it determines *which version
        of a data file is downloaded*.  A fixture added on a feature branch and
        exercised by a test on that branch exists only on that branch; if the
        branch cannot be resolved, the URL points at
        :data:`__DC3_DEFAULT_BRANCH__`, the file is not there, and the test
        fails with a download error that looks nothing like its actual cause.
        This is why ``pygit2`` is a test requirement even though it is optional
        for a general user.

    Returns
    -------
    str
        Branch name, or the commit hash when ``HEAD`` is detached (as it is for
        pull-request builds).  Defaults to :data:`__DC3_DEFAULT_BRANCH__` for
        development versions when the repository cannot be identified, and to
        the version string otherwise.
    """
    repo = git_repo()
    if repo is None:
        return __DC3_DEFAULT_BRANCH__ if '.dev' in __version__ else __version__
    return str(repo.head.target) if repo.head_is_detached else str(repo.head.shorthand)


def git_remote_path():
    """
    The main path to the GitHub repository.

    Defaults to :data:`__DC3_REPO_PATH__` if the repository cannot be identified
    (see :func:`git_repo`) or if the "origin" remote URL cannot be determined.
    Resolving this from the local remote means a fork's CI fetches data from the
    fork, not from the upstream repository.

    Returns
    -------
    str
        Remote repository path, as ``owner/name``.
    """
    repo = git_repo()
    if repo is None:
        return __DC3_REPO_PATH__
    try:
        url = repo.remotes['origin'].url
    except KeyError:
        return __DC3_REPO_PATH__
    return urlparse(url).path.replace('.git', '').removeprefix('/')


def _build_remote_url(f_name, f_type):
    """
    Build the remote URL for the :mod:`astropy.utils.data` functions.

    This function keeps URL construction in one place; if the files move, this
    is the only place that needs to change.  Note that both the repository and
    the branch are resolved from the local checkout where possible; see
    :func:`git_branch`.

    Parameters
    ----------
    f_name : str
        The base filename to search for.
    f_type : str
        The subdirectory of ``dc3/data/`` in which to find the file (e.g.,
        ``tests`` or ``templates``).

    Returns
    -------
    str
        The URL of ``f_name`` within ``f_type`` on GitHub.
    """
    parts = (
        ['https://raw.githubusercontent.com', f'/{git_remote_path()}/', f'{git_branch()}/',
         'dc3/', 'data/']
        + [f'{p}/' for p in pathlib.Path(f_type).parts]
        + [f'{f_name}']
    )
    return reduce(lambda a, b: urljoin(a, b), parts)


def fetch_remote_file(filename, filetype, force_update=False, full_url=None, return_none=False,
                      timeout=10):
    """
    Use :mod:`astropy.utils.data` to fetch a file from the remote host or cache.

    `astropy.utils.data.download_file` looks in the local cache before
    downloading from the remote server.  The remote file can be forcibly
    re-downloaded using ``force_update``.

    Parameters
    ----------
    filename : str
        The base filename to search for.
    filetype : str
        The subdirectory of ``dc3/data/`` in which to find the file (e.g.,
        ``tests`` or ``templates``).
    force_update : bool, optional
        Force `astropy.utils.data.download_file` to update the cache by
        downloading the latest version.
    full_url : str, optional
        The full URL.  If None, use :func:`_build_remote_url`.
    return_none : bool, optional
        Return None if the file cannot be found, instead of raising an error.
    timeout : int, optional
        Timeout in seconds for the download.

    Returns
    -------
    `pathlib.Path`, None
        The local path to the file in the cache, or None; see ``return_none``.

    Raises
    ------
    DC3Error
        Raised if the download fails and ``return_none`` is False.
    """
    remote_url = _build_remote_url(filename, filetype) if full_url is None else full_url

    try:
        cache_fn = astropy.utils.data.download_file(
            remote_url,
            timeout=timeout,
            cache="update" if force_update else True,
            pkgname="dc3",
        )
    except urllib.error.URLError as error:
        if return_none:
            return None
        raise DC3Error(
            f'Error downloading {filename}: {error}\n'
            f'URL attempted: {remote_url}\n'
            'If the error relates to the server not being found, check your internet connection.  '
            'If the file was recently added on a branch, check that the branch in the URL above '
            'is the one you expect; see dc3.pkg.cache.git_branch.'
        ) from error
    except TimeoutError as error:
        if return_none:
            return None
        raise DC3Error(f'Timeout error encountered: {error}') from error

    return pathlib.Path(cache_fn).resolve()


def search_cache(pattern, path_only=True):
    """
    Search the cache for items matching a pattern string.

    Parameters
    ----------
    pattern : str, None
        The pattern to match within the source URL.  If None, the full contents
        of the cache are returned; note that this is not very useful with
        ``path_only=True``, given the abstraction of the cached file names.
    path_only : bool, optional
        Only return the path(s) to the files found in the cache.  If False, a
        dictionary is returned, keyed by source URL.

    Returns
    -------
    list, dict
        The local paths of the matching objects, or a dictionary mapping source
        URL to local path; see ``path_only``.
    """
    contents = astropy.utils.data.cache_contents(pkgname="dc3")
    contents = {k: pathlib.Path(v) for k, v in contents.items() if pattern is None or pattern in k}
    return list(contents.values()) if path_only else contents


def write_file_to_cache(filename, cachename, filetype):
    """
    Use :mod:`astropy.utils.data` to save a local file to the cache.

    This writes a local file into the dc3 cache as if it had come from the
    remote server, which allows a locally created or separately downloaded file
    to stand in for the distributed version.

    Parameters
    ----------
    filename : str
        The filename of the local file to save.
    cachename : str
        The name of the cached version of the file.
    filetype : str
        The subdirectory of ``dc3/data/`` in which the file belongs.
    """
    # Build the url_key as if this file were in the remote location
    url_key = _build_remote_url(cachename, filetype)
    astropy.utils.data.import_file_to_cache(url_key, filename, pkgname="dc3")


def remove_from_cache(cache_url=None, pattern=None, allow_multiple=False):
    """
    Remove a previously downloaded file from the dc3 cache.

    The file is specified either by its full URL or by a pattern used to search
    the cache.

    Parameters
    ----------
    cache_url : list, str, optional
        One or more URLs in the cache to be deleted, if they exist.  If
        ``allow_multiple`` is False, this must be a single string.
    pattern : str, optional
        A pattern used to search the cache for the relevant file(s).  If
        ``allow_multiple`` is False, this must match a single file, otherwise a
        warning is issued and nothing is deleted.
    allow_multiple : bool, optional
        If the search pattern yields multiple results, remove all of them.
    """
    if cache_url is None:
        _url = search_cache(pattern, path_only=False)
        if len(_url) == 0:
            warnings.warn(f'Cache does not include a file matching the pattern {pattern}.')
            return
        _url = list(_url.keys())
    elif not isinstance(cache_url, list):
        _url = [cache_url]
    else:
        _url = cache_url

    if len(_url) > 1 and not allow_multiple:
        warnings.warn(
            'Function found or was provided with multiple entries to be removed.  Either set '
            'allow_multiple=True, or try again with a single url or a more specific pattern.  '
            f'URLs passed/found are: {_url}'
        )
        return

    for u in _url:
        astropy.utils.data.clear_download_cache(hashorurl=u, pkgname='dc3')
