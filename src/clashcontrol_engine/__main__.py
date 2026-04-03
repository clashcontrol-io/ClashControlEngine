"""Allow running with `python -m clashcontrol_engine`."""
import multiprocessing
multiprocessing.freeze_support()

from clashcontrol_engine.cli import main

main()
