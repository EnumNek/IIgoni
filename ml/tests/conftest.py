"""Общие фикстуры тестов ml/. Тесты не пишут в ml/data, ml/weights, ml/reports."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.stub_network import build_grid_python  # noqa: E402


@pytest.fixture(scope="session")
def grid_network() -> dict:
    return build_grid_python(4)


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory, grid_network) -> Path:
    """Маленький синтетический датасет (решётка 4×4, 60 сценариев)."""
    from ml.synthetic import generate

    out = tmp_path_factory.mktemp("synth")
    generate(grid_network, out, n_scenarios=60, n_base=3, seed=1, workers=1)
    return out


@pytest.fixture(scope="session")
def trained_weights(tmp_path_factory, tiny_dataset) -> Path:
    """Быстрое обучение (--quick) на маленьком датасете — проверка всего конвейера."""
    from ml import train

    tmp = tmp_path_factory.mktemp("run")
    train.main(["--quick", "--source", "synthetic", "--synth-dir", str(tiny_dataset),
                "--weights-dir", str(tmp / "weights"), "--reports-dir", str(tmp / "reports"),
                "--cache-path", str(tmp / "cache.npz"), "--skip-benchmark"])
    return tmp


def load_json(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))
