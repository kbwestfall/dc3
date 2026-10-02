"""
Sphinx configuration for the dc3 documentation.

Structured after PypeIt's ``doc/conf.py``, with one deliberate difference:
docstrings are NumPy style and rendered by ``numpydoc``, not by
``sphinx.ext.napoleon``.  ``numpydoc`` is the renderer numpy, scipy and astropy
use, and it can validate docstrings as well as render them.

Build with ``make html`` from this directory; see the Makefile.  Read the Docs
builds from the committed files and runs none of the Makefile's generation
steps, so the API pages (``make apirst``) and the characterization figures
(``make figures``) must be regenerated and committed when they change.
"""

from importlib import metadata
from pathlib import Path
import tomllib


# -- Project information -----------------------------------------------------

def get_package_version():
    """
    Return the installed version of the package named in ``pyproject.toml``.

    Returns
    -------
    str
        The version.
    """
    project_file = Path(__file__).resolve().parents[1] / 'pyproject.toml'
    with open(project_file, 'rb') as f:
        return metadata.version(tomllib.load(f)['project']['name'])


project = 'dc3'
author = 'Kyle B. Westfall'
copyright = '2026, Kyle B. Westfall'
version = get_package_version()
release = version


# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx_design',
]

root_doc = 'index'
source_suffix = '.rst'
# The release notes are included into whatsnew.rst, as in PypeIt, rather than
# built as pages of their own.
exclude_patterns = [
    '_build', 'include/*.rst', 'releases/*.rst', 'scripts', 'figures', 'sphinx_warnings.out'
]
templates_path = []

pygments_style = 'sphinx'
todo_include_todos = True


# -- numpydoc ------------------------------------------------------------------

# The API pages written by sphinx-apidoc already document every member with
# autodoc; numpydoc listing them again would duplicate each one.
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False


# -- autodoc -------------------------------------------------------------------

autodoc_member_order = 'bysource'


# -- intersphinx ---------------------------------------------------------------

# ppxf publishes no object inventory, so it cannot be listed here; see
# include/links.rst.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'pydantic': ('https://docs.pydantic.dev/latest/', None),
    'specutils': ('https://specutils.readthedocs.io/en/stable/', None),
}


# -- HTML output ---------------------------------------------------------------

html_theme = 'pydata_sphinx_theme'
#html_theme_options = {
#    'prev_next_buttons_location': None,
#}
htmlhelp_basename = 'dc3doc'
