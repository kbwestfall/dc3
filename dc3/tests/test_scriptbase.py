"""
Tests for :class:`~dc3.scripts.scriptbase.ScriptBase`.
"""

import argparse
import logging
from pathlib import Path

import pytest

from dc3.par.dc3par import DC3Par
from dc3.scripts.scriptbase import ScriptBase, configure_matplotlib
from dc3.scripts.version import Version


class ExampleScript(ScriptBase):
    """A minimal script used to exercise the base class."""

    @classmethod
    def get_parser(cls, width=None):
        """Build a parser with a config option and one override."""
        parser = super().get_parser(description='An example.', width=width)
        cls.add_config_argument(parser, parset_class=DC3Par)
        parser.add_argument('--ncpu', type=int, default=None, help='Number of processes.')
        parser.add_argument(
            '--outdir', type=str, default='current working directory', help='Output directory.'
        )
        return parser

    @classmethod
    def main(cls, args):
        """Return the arguments, so a test can see them."""
        return args


# ----------------------------------------------------------------------
# Naming and parsing
# ----------------------------------------------------------------------
def test_script_name_follows_the_module():
    """The script name is the module name with the dc3 prefix."""
    assert Version.name() == 'dc3_version', \
        'Script name does not match the registered entry-point name'


def test_parse_args():
    """Arguments parse, and the logging options are present by default."""
    args = ExampleScript.parse_args(['--ncpu', '4'])
    assert args.ncpu == 4, 'Command-line value was not parsed'
    assert args.verbosity == 1, 'Default verbosity changed'
    assert args.log_file is None, 'A log file is written by default, which it should not be'


def test_cwd_placeholder_is_replaced():
    """
    The 'current working directory' placeholder resolves at parse time.

    The placeholder exists so that generated help documentation does not bake in
    the directory of whoever last built the docs.
    """
    args = ExampleScript.parse_args([])
    assert args.outdir == str(Path.cwd()), \
        'The placeholder default was not replaced with the working directory'
    # The help text, however, is generated from the unresolved parser
    assert 'current working directory' in ExampleScript.get_parser().format_help(), \
        'The placeholder should survive in the help text, for reproducible docs'


def test_log_options_can_be_omitted():
    """A script that does no logging need not carry the logging options."""
    parser = ScriptBase.get_parser(description='x', include_log_options=False)
    assert '--verbosity' not in parser.format_help(), \
        'Logging options were added despite being suppressed'


# ----------------------------------------------------------------------
# Parameter resolution
# ----------------------------------------------------------------------
def test_resolve_par_defaults():
    """With no file and no overrides, the parameter set takes its defaults."""
    par = ExampleScript.resolve_par(DC3Par)
    assert par.ncpu == 1, 'Default was not used when nothing else was supplied'


def test_resolve_par_layers_file_and_cli(tmp_path):
    """Command line beats file beats default, including within subsections."""
    f = tmp_path / 'dc3.toml'
    DC3Par(ncpu=4, template={'velscale_ratio': 2}).to_toml(cfg_file=f)

    par = ExampleScript.resolve_par(
        DC3Par, config_file=f, overrides={'ncpu': 8, 'template': {'epsilon_sigma': None}}
    )
    assert par.ncpu == 8, 'Command-line value did not override the file'
    assert par.template.velscale_ratio == 2, 'File value was lost'
    assert par.template.epsilon_sigma == 0.1, \
        'An unset (None) override did not fall through to the file or default'


def test_resolve_par_reads_the_file_without_validating_it(tmp_path):
    """
    A file that is only valid once combined with the command line is accepted.

    Validating the file on its own would reject a configuration whose missing
    half arrives from the command line, which is the normal way of using both.
    """
    f = tmp_path / 'partial.toml'
    # Gauss-Hermite moments with a pedestal is invalid; the command line fixes it
    f.write_text('[dc3]\n[dc3.fit]\nmoments = 4\n[dc3.template]\nsigma_floor = 5.0\n')

    par = ExampleScript.resolve_par(
        DC3Par, config_file=f, overrides={'template': {'sigma_floor': 0.0}}
    )
    assert par.fit.moments == 4, 'The file value was lost'
    assert par.template.sigma_floor == 0.0, 'The command-line correction was not applied'


def test_resolve_par_still_validates_the_result(tmp_path):
    """The combined configuration is validated, even though the file alone is not."""
    f = tmp_path / 'bad.toml'
    f.write_text('[dc3]\n[dc3.fit]\nmoments = 4\n[dc3.template]\nsigma_floor = 5.0\n')
    with pytest.raises(Exception, match='matched template resolution'):
        ExampleScript.resolve_par(DC3Par, config_file=f)


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    'verbosity,expected',
    [(0, logging.WARNING), (1, logging.INFO), (2, logging.DEBUG)]
)
def test_verbosity_maps_to_logging_level(verbosity, expected):
    """Each verbosity level maps to the documented logging level."""
    from dc3 import log
    assert log.convert_verbosity_to_logging_level(verbosity) == expected, \
        f'Verbosity {verbosity} did not map to the documented logging level'


def test_invalid_verbosity_is_rejected():
    """A verbosity outside 0-2 is an error rather than a silent clamp."""
    from dc3 import log
    with pytest.raises(ValueError, match='Verbosity level'):
        log.convert_verbosity_to_logging_level(3)


def test_default_log_file_name():
    """The default log file names the script and the time it ran."""
    name = Version.default_log_file()
    assert name.startswith('dc3_version_'), 'Log file name does not identify the script'
    assert name.endswith('.log'), 'Log file name does not end in .log'


def test_init_log_writes_a_file(tmp_path):
    """Naming a log file produces one, and it records what was logged."""
    from dc3 import log
    f = tmp_path / 'run.log'
    args = argparse.Namespace(verbosity=1, log_file=str(f), log_level=None)
    try:
        ScriptBase.init_log(args)
        log.info('a message that should reach the file')
        log.close_file()
        assert f.exists(), 'No log file was written despite one being named'
        assert 'a message that should reach the file' in f.read_text(), \
            'The log file does not contain the logged message'
    finally:
        # Restore the console-only logger for the rest of the suite
        log.init(level=logging.INFO)


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def test_expandpath_handles_wildcards_in_directories(tmp_path):
    """
    Wildcards may appear in directory components, not only in the file name.

    This is what it offers over glob.glob.
    """
    for sub in ['run1', 'run2']:
        (tmp_path / sub).mkdir()
        (tmp_path / sub / 'spec.fits').touch()
    found = sorted(p.name for p in ScriptBase.expandpath(str(tmp_path / 'run*' / '*.fits')))
    assert found == ['spec.fits', 'spec.fits'], \
        'A wildcard in a directory component was not expanded'


def test_configure_matplotlib_forces_a_headless_backend():
    """Without --show, a non-interactive backend is selected."""
    import matplotlib
    original = matplotlib.get_backend()
    try:
        configure_matplotlib(False)
        assert matplotlib.get_backend().lower() == 'agg', \
            'A headless backend was not selected, so plotting would need a display'
        configure_matplotlib(True)
        assert matplotlib.get_backend().lower() == 'agg', \
            'configure_matplotlib(True) should leave the backend alone, not change it'
    finally:
        matplotlib.use(original, force=True)


# ----------------------------------------------------------------------
# The version script
# ----------------------------------------------------------------------
def test_version_script(capsys):
    """The version script reports the package version."""
    import dc3
    Version.main(Version.parse_args([]))
    assert dc3.__version__ in capsys.readouterr().out, \
        'The version script did not report the package version'


def test_version_script_dependencies(capsys):
    """With --dependencies, the versions of the key dependencies are reported."""
    Version.main(Version.parse_args(['--dependencies']))
    out = capsys.readouterr().out
    for package in ['numpy', 'ppxf', 'astropy']:
        assert package in out, f'{package} was not reported among the dependencies'
