"""Extraction de caractéristiques par cycle à partir des signaux bruts.

Chaque cycle dure 60 s : PS2 contient 6000 points (100 Hz) et FS1 600 points
(10 Hz). Un modèle classique ne peut pas exploiter directement 6600 colonnes
fortement corrélées ; on résume donc chaque signal par des statistiques
globales, des statistiques sur sa dérivée, des moyennes par fenêtre temporelle
(le retard de commutation de la valve décale les transitoires dans le temps)
et l'énergie dans quelques bandes de fréquence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

N_WINDOWS = 10
N_BANDS = 4


def _global_stats(x: np.ndarray) -> dict[str, np.ndarray]:
    q05, q25, q50, q75, q95 = np.percentile(x, [5, 25, 50, 75, 95], axis=1)
    return {
        "mean": x.mean(axis=1),
        "std": x.std(axis=1),
        "min": x.min(axis=1),
        "max": x.max(axis=1),
        "median": q50,
        "q05": q05,
        "q25": q25,
        "q75": q75,
        "q95": q95,
        "iqr": q75 - q25,
        "ptp": x.max(axis=1) - x.min(axis=1),
        "rms": np.sqrt((x ** 2).mean(axis=1)),
        "skew": np.nan_to_num(stats.skew(x, axis=1)),
        "kurtosis": np.nan_to_num(stats.kurtosis(x, axis=1)),
        "argmax_s": x.argmax(axis=1) / x.shape[1] * 60.0,
        "argmin_s": x.argmin(axis=1) / x.shape[1] * 60.0,
    }


def _derivative_stats(x: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    d = np.diff(x, axis=1) * fs  # unité / s
    return {
        "d_absmean": np.abs(d).mean(axis=1),
        "d_std": d.std(axis=1),
        "d_max": d.max(axis=1),
        "d_min": d.min(axis=1),
    }


def _window_stats(x: np.ndarray, n_windows: int) -> dict[str, np.ndarray]:
    out = {}
    for i, w in enumerate(np.array_split(x, n_windows, axis=1)):
        out[f"w{i}_mean"] = w.mean(axis=1)
        out[f"w{i}_std"] = w.std(axis=1)
    return out


def _spectral_stats(x: np.ndarray, n_bands: int) -> dict[str, np.ndarray]:
    xc = x - x.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(xc, axis=1)) ** 2
    power = power[:, 1:]  # on enlève la composante continue
    total = power.sum(axis=1) + 1e-12
    # bandes logarithmiques : les basses fréquences portent l'essentiel
    n = power.shape[1]
    inner = np.geomspace(1, n, n_bands + 1).astype(int)[1:-1]
    edges = np.unique(np.r_[0, inner, n])
    out = {}
    for i in range(len(edges) - 1):
        out[f"band{i}_ratio"] = power[:, edges[i]:edges[i + 1]].sum(axis=1) / total
    out["spec_log_energy"] = np.log1p(total)
    return out


def extract_signal_features(x: np.ndarray, name: str, fs: float,
                            n_windows: int = N_WINDOWS,
                            n_bands: int = N_BANDS) -> pd.DataFrame:
    """Calcule toutes les caractéristiques d'un capteur (matrice cycles x points)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    feats = {}
    feats.update(_global_stats(x))
    feats.update(_derivative_stats(x, fs))
    feats.update(_window_stats(x, n_windows))
    feats.update(_spectral_stats(x, n_bands))
    df = pd.DataFrame(feats)
    df.columns = [f"{name}_{c}" for c in df.columns]
    return df


def build_feature_matrix(ps2: np.ndarray, fs1: np.ndarray) -> pd.DataFrame:
    """Matrice de caractéristiques complète (une ligne par cycle)."""
    X = pd.concat(
        [
            extract_signal_features(ps2, "PS2", fs=100.0),
            extract_signal_features(fs1, "FS1", fs=10.0),
        ],
        axis=1,
    )
    X.index = pd.RangeIndex(1, len(X) + 1, name="cycle")
    return X
