"""
Base class for the dc3 command-line scripts.

.. note::

    Adapted from ``pypeit/scripts/scriptbase.py`` in `PypeIt
    <https://github.com/pypeit/PypeIt>`__ (BSD 3-Clause); see
    ``licenses/README.rst``.  :class:`ScriptBase` additionally knows how to
    resolve a parameter set from a configuration file layered with command-line
    overrides, which PypeIt handles elsewhere.

Writing a script
----------------

A script subclasses :class:`ScriptBase`, overrides :func:`ScriptBase.get_parser`
and :func:`ScriptBase.main`, and is registered in ``pyproject.toml`` as
``<name> = "dc3.scripts.<module>:<Class>.entry_point"``.

.. important::

    **Keep heavy imports inside** :func:`ScriptBase.main`, not at module scope.
    Every registered entry point is imported when the console scripts are
    resolved, so a module-level ``import`` of the fitting machinery makes
    ``--help`` pay for it.

.. include:: ../include/links.rst
"""

import argparse
import datetime
from pathlib import Path

from .. import log


__all__ = ['ScriptBase', 'configure_matplotlib']


class ScriptBase:
    """
    Base class for all dc3 command-line scripts.

    Subclasses override :func:`get_parser` and :func:`main`.
    """

    @classmethod
    def entry_point(cls):
        """Run the script.  This is what ``pyproject.toml`` registers."""
        cls.main(cls.parse_args())

    @classmethod
    def name(cls):
        """
        Return the name of the script.

        By default this is the module name with ``dc3_`` prepended, which is the
        convention the registered entry points follow.

        Returns
        -------
        str
            The script name.
        """
        return f"dc3_{cls.__module__.split('.')[-1]}"

    @classmethod
    def parse_args(cls, options=None):
        """
        Parse the command-line arguments.

        Parameters
        ----------
        options : list, optional
            Arguments to parse.  If None, the command line is used.

        Returns
        -------
        :class:`argparse.Namespace`
            The parsed arguments.
        """
        parser = cls.get_parser()
        cls._fill_parser_cwd(parser)
        return parser.parse_args() if options is None else parser.parse_args(options)

    @staticmethod
    def _fill_parser_cwd(parser):
        """
        Replace the placeholder default ``'current working directory'``.

        Any action whose default is exactly that string has it replaced with the
        actual working directory.  The placeholder exists so that the generated
        help documentation does not bake in the directory of whoever last built
        the docs.  The ``parser`` is edited **in place**.

        Parameters
        ----------
        parser : :class:`argparse.ArgumentParser`
            The parser to edit.
        """
        for action in parser._actions:
            if action.default == 'current working directory':
                action.default = str(Path.cwd())

    @classmethod
    def get_parser(
        cls, description=None, width=None, formatter=argparse.ArgumentDefaultsHelpFormatter,
        include_log_options=True
    ):
        """
        Construct the command-line argument parser.

        Subclasses override this, calling it to build the parser and then adding
        their own arguments.

        .. warning::

            Any argument defaulting to the string ``'current working
            directory'`` has that replaced by the actual working directory when
            the script runs; see :func:`_fill_parser_cwd`.

        Parameters
        ----------
        description : str, optional
            A short description of what the script does.
        width : int, optional
            Maximum width of the formatted help output.  If None, the terminal
            width is used.
        formatter : :class:`argparse.HelpFormatter`, optional
            Class used to format the help output.
        include_log_options : bool, optional
            Include the options controlling logging.

        Returns
        -------
        :class:`argparse.ArgumentParser`
            The command-line interpreter.
        """
        parser = argparse.ArgumentParser(
            description=description, formatter_class=lambda prog: formatter(prog, width=width)
        )
        if not include_log_options:
            return parser
        parser.add_argument(
            '-v', '--verbosity', type=int, default=1,
            help='Verbosity level, which must be 0, 1, or 2.  Level 0 includes warning and '
                 'error messages, level 1 adds informational messages, and level 2 adds '
                 'debugging messages and the calling sequence.'
        )
        parser.add_argument(
            '--log_file', type=str, default=None,
            help='Name for the log file.  If set to "default", a name is generated from the '
                 'script name and the current time.  If not set, no log file is written.'
        )
        parser.add_argument(
            '--log_level', type=int, default=None,
            help='Verbosity level for the log file.  If a log file is written and this is not '
                 'set, it matches the console verbosity.'
        )
        return parser

    @staticmethod
    def add_config_argument(parser, parset_class=None):
        """
        Add the option naming a TOML configuration file.

        Parameters
        ----------
        parser : :class:`argparse.ArgumentParser`
            The parser to add the argument to.
        parset_class : type, optional
            The :class:`~dc3.par.parset.ParSet` subclass the file configures,
            used only to name its section in the help text.
        """
        section = (
            '' if parset_class is None
            else f'  Reads the [{parset_class.default_key}] section.'
        )
        parser.add_argument(
            '-c', '--config', type=str, default=None,
            help=f'TOML configuration file.{section}  Any value given on the command line '
                 'overrides the file, and anything set in neither takes its default.'
        )

    @classmethod
    def resolve_par(cls, parset_class, config_file=None, overrides=None):
        """
        Build a parameter set from a configuration file and command-line values.

        Precedence runs defaults < file < command line.  The file is read but
        *not* validated on its own, since a configuration that is incomplete
        without its command-line half is not an error.

        Parameters
        ----------
        parset_class : type
            The :class:`~dc3.par.parset.ParSet` subclass to build.
        config_file : str, :class:`pathlib.Path`, optional
            The configuration file.  If None, only the defaults and the
            command-line values are used.
        overrides : dict, optional
            Values supplied on the command line, nested to match the parameter
            set.  A value of None means "not supplied" and does not override.

        Returns
        -------
        ParSet
            The resolved parameter set.
        """
        file_cfg = None if config_file is None else parset_class.config_dict(config_file)
        return parset_class.from_layers(file_cfg=file_cfg, cli_cfg=overrides)

    @classmethod
    def init_log(cls, args):
        """
        Initialize the logger from the parsed command-line arguments.

        Parameters
        ----------
        args : :class:`argparse.Namespace`
            The parsed arguments, carrying ``verbosity``, ``log_file`` and
            ``log_level``.
        """
        level = log.convert_verbosity_to_logging_level(args.verbosity)
        log_file_level = (
            None if args.log_level is None
            else log.convert_verbosity_to_logging_level(args.log_level)
        )
        if args.log_file == 'default':
            _log_file = cls.default_log_file()
        elif args.log_file in ['None', None]:
            _log_file = None
        else:
            _log_file = args.log_file
        log.init(level=level, log_file=_log_file, log_file_level=log_file_level)

    @classmethod
    def default_log_file(cls):
        """
        Return the default name for the log file.

        Returns
        -------
        str
            The script name and a UT timestamp, to the minute.
        """
        timestamp = datetime.datetime.now(datetime.UTC).strftime('%Y%m%d-%H%M')
        return f'{cls.name()}_{timestamp}.log'

    @classmethod
    def main(cls, args):
        """
        Execute the script.  Subclasses override this.

        Parameters
        ----------
        args : :class:`argparse.Namespace`
            The parsed arguments.
        """
        pass

    @staticmethod
    def expandpath(path_pattern):
        """
        Expand a path pattern with wildcards into all matching files.

        Unlike :func:`glob.glob`, wildcards may occur anywhere in the path,
        including in directory components.

        Parameters
        ----------
        path_pattern : str
            Search pattern for files on disk.

        Returns
        -------
        generator
            The matching paths.
        """
        p = Path(path_pattern).expanduser()
        parts = p.parts[p.is_absolute():]
        return Path(p.root).glob(str(Path(*parts)))


def configure_matplotlib(show):
    """
    Select a matplotlib backend appropriate to how the script was invoked.

    An interactive backend is only needed when the user asked to see something.
    Otherwise force a non-interactive one *before* anything imports
    :mod:`matplotlib.pyplot`, so that quality-assessment figures can be written
    with no display attached -- which is the normal case for a production run
    and for continuous integration.

    Parameters
    ----------
    show : bool
        If True, leave the configured backend alone.
    """
    if show:
        return
    # NOTE: Deferred deliberately, for the same reason a script's imports belong
    # in its main: this module is imported by every registered entry point, so a
    # module-level matplotlib import would make even `--help` pay for it.
    import matplotlib
    matplotlib.use('Agg', force=True)
