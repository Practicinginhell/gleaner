"""Command-line entry point: `python -m gleaner`."""

import sys
from collections.abc import Sequence

from gleaner import __version__
lazy from gleaner import cli

VERSION_FLAGS = ("--version", "-V")


def main(argv: Sequence[str] | None = None) -> int:
    """Run gleaner's CLI, or print the version for a leading `--version` / `-V`.

    Args:
        argv: Command-line arguments without the program name; defaults to `sys.argv[1:]`.

    Returns:
        The process exit code.
    """
    argv = sys.argv[1:] if argv is None else list(argv)

    # Only a leading flag asks for the version; later arguments belong to the app's commands.
    if argv[:1] and argv[0] in VERSION_FLAGS:
        print(f"gleaner {__version__}")
        return 0

    return cli.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
