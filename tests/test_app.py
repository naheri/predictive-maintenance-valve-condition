"""Tests de l'application web avec un petit modèle entraîné à la volée."""
import importlib
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from valve_ml.models import make_pipeline  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n = 30
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"), index=pd.RangeIndex(1, n + 1, name="cycle"))
    valve = np.where(X["a"] > 0, 100, 80)
    X["a"] += np.where(valve == 100, 2, -2)
    model = make_pipeline(LogisticRegression(), k="all").fit(X, (valve == 100).astype(int))
    joblib.dump({"model": model, "name": "Test", "features": list(X.columns), "n_train": 20}, tmp_path / "modele_valve.joblib")
    X.to_csv(tmp_path / "features.csv.gz")
    pd.DataFrame({"valve": valve}, index=X.index).to_csv(tmp_path / "profile.csv")
    monkeypatch.setenv("VALVE_MODEL_DIR", str(tmp_path))
    sys.path.insert(0, str(ROOT / "app"))
    import app as app_module
    app_module = importlib.reload(app_module)
    return app_module.app.test_client()


def test_api_predict(client):
    r = client.get("/api/predict?cycle=25")
    assert r.status_code == 200
    data = r.get_json()
    assert data["jeu"] == "test" and 0 <= data["proba_optimale"] <= 1
    assert data["prediction"] in {"optimale", "non optimale"}


def test_api_errors(client):
    assert client.get("/api/predict?cycle=999").status_code == 400
    assert client.get("/api/predict?cycle=abc").status_code == 400


def test_index_page(client):
    r = client.get("/?cycle=3")
    assert r.status_code == 200 and "Valve" in r.get_data(as_text=True)
