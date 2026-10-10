import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantdesk.config import load_instruments, load_settings, load_venues  # noqa: E402
from quantdesk.synthetic import make_store  # noqa: E402


@pytest.fixture(scope="session")
def instruments():
    return load_instruments()


@pytest.fixture(scope="session")
def venues():
    return load_venues()


@pytest.fixture()
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def synth():
    return make_store(start="2014-01-01", n_stocks=40, seed=3)
