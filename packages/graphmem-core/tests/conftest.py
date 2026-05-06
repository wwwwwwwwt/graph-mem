import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def tmp_home():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)
