"""Allow running with `python -m clashcontrol_engine`."""
import multiprocessing
multiprocessing.freeze_support()

from clashcontrol_engine._bootstrap import configure_io
configure_io()

from clashcontrol_engine.cli import main

main()
