"""
Report the dc3 version and the versions of its key dependencies.
"""

from .scriptbase import ScriptBase


class Version(ScriptBase):
    """Report version information."""

    @classmethod
    def get_parser(cls, width=None):
        """
        Construct the command-line argument parser.

        Parameters
        ----------
        width : int, optional
            Maximum width of the formatted help output.

        Returns
        -------
        :class:`argparse.ArgumentParser`
            The command-line interpreter.
        """
        parser = super().get_parser(
            description='Report the dc3 version and the versions of its key dependencies.',
            width=width, include_log_options=False
        )
        parser.add_argument(
            '--dependencies', default=False, action='store_true',
            help='Also report the versions of the packages dc3 depends on.  ppxf is worth '
                 'checking against: two of its functions are on the critical path.'
        )
        return parser

    @classmethod
    def main(cls, args):
        """
        Execute the script.

        Parameters
        ----------
        args : :class:`argparse.Namespace`
            The parsed arguments.
        """
        # NOTE: Imports are deliberately inside main; see the module
        # documentation for dc3.scripts.scriptbase.
        import importlib.metadata

        import dc3

        print(f'dc3 {dc3.__version__}')
        if not args.dependencies:
            return

        packages = [
            'numpy', 'scipy', 'astropy', 'matplotlib', 'specutils', 'ppxf', 'pydantic'
        ]
        for package in packages:
            try:
                print(f'  {package:<12} {importlib.metadata.version(package)}')
            except importlib.metadata.PackageNotFoundError:
                print(f'  {package:<12} not installed')
