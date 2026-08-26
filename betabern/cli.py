"""Unified command-line interface for betabern.

Exposes each step as a subcommand of a single ``betabern`` entry point, e.g. ``betabern fit ...``
or ``betabern benchmark ...``. Command modules are imported lazily so that ``betabern`` and
``betabern --help`` stay cheap (no torch / Lightning import) and a single command with heavy or
broken dependencies cannot break the whole CLI.
"""

import argparse
import importlib
import sys

from betabern import __version__

# (subcommand, module path, one-line help). Each module must expose
# ``add_arguments(parser)`` and ``main(args)``; its docstring is the subcommand's long help.
COMMANDS = [
    ("fit", "betabern.commands.fit",
     "Fit an estimator to a response log or IRW table; write an ability-annotated CSV"),
    ("simulate", "betabern.commands.simulate",
     "Generate a simulated IRT response log (with ground truth) as a CSV"),
    ("fetch-irw", "betabern.commands.fetch",
     "Download + cache Item Response Warehouse tables (needs the 'irw' package)"),
    ("benchmark", "betabern.commands.benchmark",
     "Run a model-comparison benchmark config; write runs/summary/paired CSVs + report"),
    ("report", "betabern.commands.report",
     "Re-render the Markdown report for a benchmark result dir"),
    ("merge", "betabern.commands.merge",
     "Merge benchmark result dirs with disjoint datasets"),
]

_MODULES = {name: module for name, module, _ in COMMANDS}


def _build_top_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="betabern",
        description="Beta-Bernstein Bridge: closed-form Bayesian ability measurement.",
    )
    parser.add_argument("-V", "--version", action="version",
                        version=f"betabern {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    for name, _, help_text in COMMANDS:
        # Listed for `betabern --help`; real arguments are added lazily below.
        sub.add_parser(name, help=help_text, add_help=False)
    return parser


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    top_parser = _build_top_parser()

    if not argv or argv[0] in ("-h", "--help"):
        top_parser.print_help()
        return
    if argv[0] in ("-V", "--version"):
        top_parser.parse_args(argv)
        return

    command = argv[0]
    module_path = _MODULES.get(command)
    if module_path is None:
        top_parser.error(f"invalid command: {command!r} "
                         f"(choose from {', '.join(_MODULES)})")

    module = importlib.import_module(module_path)
    cmd_parser = argparse.ArgumentParser(prog=f"betabern {command}", description=module.__doc__,
                                         formatter_class=argparse.RawDescriptionHelpFormatter)
    module.add_arguments(cmd_parser)
    args = cmd_parser.parse_args(argv[1:])
    module.main(args)


if __name__ == "__main__":
    main()
