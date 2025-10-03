from pathlib import Path
import pytest

@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).parent.parent / "test_data"
