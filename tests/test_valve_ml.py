"""Tests unitaires (pytest) : données, extraction de caractéristiques, sélection, pipeline.

Ils n'ont pas besoin des vrais fichiers : les signaux sont simulés.
Lancement : python -m pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from valve_ml.data import N_TRAIN, load_profile, load_sensor, make_target, train_test_split_by_cycle  # noqa: E402
from valve_ml.features import N_WINDOWS, build_feature_matrix, extract_signal_features  # noqa: E402
from valve_ml.models import all_pipelines, make_pipeline  # noqa: E402
from valve_ml.selection import CorrelationFilter  # noqa: E402


@pytest.fixture
def signals():
    rng = np.random.default_rng(0)
    n = 40
    ps2 = 100 + rng.normal(0, 5, (n, 6000))
    fs1 = 8 + rng.normal(0, 1, (n, 600))
    return ps2, fs1


@pytest.fixture
def profile_file(tmp_path):
    rows = [[3, 100, 0, 130, 1], [20, 90, 1, 115, 0], [100, 73, 2, 90, 0], [100, 100, 0, 100, 0]]
    path = tmp_path / "profile.txt"
    path.write_text("\n".join("\t".join(map(str, r)) for r in rows))
    return path


def test_load_profile_and_target(profile_file):
    prof = load_profile(profile_file)
    assert list(prof.columns) == ["cooler", "valve", "pump", "accumulator", "stable"]
    assert prof.index[0] == 1  # numérotation des cycles à partir de 1
    assert make_target(prof).tolist() == [1, 0, 0, 1]


def test_load_sensor_shape(tmp_path):
    arr = np.arange(12, dtype=float).reshape(3, 4)
    path = tmp_path / "S.txt"
    np.savetxt(path, arr, delimiter="\t")
    np.testing.assert_allclose(load_sensor(path), arr)


def test_split_first_2000_cycles():
    X = pd.DataFrame({"a": np.arange(2205)})
    y = pd.Series(np.arange(2205) % 2)
    X_tr, X_te, y_tr, y_te = train_test_split_by_cycle(X, y)
    assert len(X_tr) == N_TRAIN == 2000 and len(X_te) == 205
    assert X_te["a"].iloc[0] == 2000  # aucun recouvrement


def test_signal_features_values():
    x = np.tile(np.array([0.0, 1.0, 2.0, 3.0]), (2, 25))  # 2 cycles de 100 points
    f = extract_signal_features(x, "S", fs=10.0)
    assert f.shape[0] == 2
    assert f["S_mean"].iloc[0] == pytest.approx(1.5)
    assert f["S_min"].iloc[0] == 0 and f["S_max"].iloc[0] == 3
    assert f["S_ptp"].iloc[0] == 3
    assert f["S_rms"].iloc[0] == pytest.approx(np.sqrt(3.5))
    assert sum(c.endswith("_mean") and c.startswith("S_w") for c in f.columns) == N_WINDOWS


def test_band_ratios_sum_to_one(signals):
    ps2, _ = signals
    f = extract_signal_features(ps2, "PS2", fs=100.0)
    bands = f.filter(regex=r"band\d+_ratio")
    np.testing.assert_allclose(bands.sum(axis=1), 1.0, atol=1e-6)


def test_feature_matrix(signals):
    X = build_feature_matrix(*signals)
    assert X.shape[0] == 40
    assert X.index[0] == 1
    assert not X.isna().any().any()
    assert any(c.startswith("PS2_") for c in X) and any(c.startswith("FS1_") for c in X)


def test_correlation_filter_removes_duplicates():
    rng = np.random.default_rng(1)
    a = rng.normal(size=200)
    X = pd.DataFrame({"a": a, "a_copy": 2 * a + 1, "b": rng.normal(size=200), "const": 1.0})
    cf = CorrelationFilter(0.95).fit(X)
    assert list(cf.transform(X).columns) == ["a", "b"]
    assert list(cf.get_feature_names_out()) == ["a", "b"]


def test_pipelines_fit_and_predict(signals):
    ps2, fs1 = signals
    ps2[:20] += 20  # deux classes séparables
    X = build_feature_matrix(ps2, fs1)
    y = pd.Series([1] * 20 + [0] * 20, index=X.index)
    pipe = make_pipeline(__import__("sklearn.linear_model", fromlist=["x"]).LogisticRegression(max_iter=1000), k=5)
    pipe.fit(X, y)
    assert (pipe.predict(X) == y).mean() == 1.0
    assert set(all_pipelines()) >= {"Régression logistique", "k-NN", "Arbre de décision", "Naive Bayes", "SVM (RBF)"}
