import logging
import os
import pathlib
import subprocess

logging.basicConfig(level=logging.DEBUG)


def pytest_sessionstart():
    """Pytest plugin to trigger maturin builds before running tests."""
    if os.environ.get("CI") is None:
        # Running maturin develop --skip-install builds a "linux" wheel which PyPI rejects
        # (https://peps.python.org/pep-0513/#rationale). When testing an already built wheel, we
        # neither want to rebuild nor pollute the target/wheels directory with a wheel that PyPI
        # will reject.
        working_dir = pathlib.Path(__file__).parent.parent
        subprocess.check_call(["maturin", "develop", "--skip-install"], cwd=working_dir)
