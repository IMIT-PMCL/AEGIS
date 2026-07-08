"""Minimal example: run the reference demo from Python.

Equivalent to ``aegis demo`` on the command line. Runs fully offline.
"""

from aegis.cli import main

if __name__ == "__main__":
    main(["demo", "--provenance", "example_provenance.jsonl"])
