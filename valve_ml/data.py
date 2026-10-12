"""Chargement des données brutes du banc hydraulique (UCI, Helwig et al. 2015).

Seuls les trois fichiers autorisés par les consignes sont utilisés :
PS2.txt (pression, 100 Hz), FS1.txt (débit, 10 Hz) et profile.txt (conditions).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROFILE_COLUMNS = ["cooler", "valve", "pump", "accumulator", "stable"]
N_TRAIN = 2000  # les 2000 premiers cycles servent à construire le modèle

# Dossiers explorés (dans l'ordre) pour trouver les fichiers .txt
_CANDIDATES = [
    os.environ.get("VALVE_DATA_DIR", ""),
    "data",
    "../data",
    "../Projet_ML",
    "../../Projet_ML",
    "Projet_ML",
    "/mnt/project-files/Projet_ML",
    "/content/data",  # Google Colab
]


def find_data_dir(required=("PS2.txt", "FS1.txt", "profile.txt")) -> Path:
    """Renvoie le premier dossier candidat contenant tous les fichiers requis."""
    for cand in _CANDIDATES:
        if cand and all((Path(cand) / f).exists() for f in required):
            return Path(cand).resolve()
    raise FileNotFoundError(
        f"Impossible de trouver {required}. Placez les fichiers dans ./data "
        "ou définissez la variable d'environnement VALVE_DATA_DIR."
    )


def load_sensor(path: str | Path) -> np.ndarray:
    """Charge une matrice capteur (lignes = cycles, colonnes = points du cycle)."""
    return pd.read_csv(path, sep="\t", header=None).to_numpy(dtype=float)


def load_profile(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=PROFILE_COLUMNS)
    df.index = pd.RangeIndex(1, len(df) + 1, name="cycle")
    return df


def make_target(profile: pd.DataFrame) -> pd.Series:
    """Cible binaire : 1 si la valve est optimale (=100 %), 0 sinon."""
    return (profile["valve"] == 100).astype(int).rename("valve_optimal")


def load_all(data_dir: str | Path | None = None):
    """Charge PS2, FS1 et profile. Renvoie (ps2, fs1, profile)."""
    data_dir = Path(data_dir) if data_dir else find_data_dir()
    ps2 = load_sensor(data_dir / "PS2.txt")
    fs1 = load_sensor(data_dir / "FS1.txt")
    profile = load_profile(data_dir / "profile.txt")
    if not (len(ps2) == len(fs1) == len(profile)):
        raise ValueError("Les fichiers n'ont pas le même nombre de cycles.")
    return ps2, fs1, profile


def train_test_split_by_cycle(X, y, n_train: int = N_TRAIN):
    """Découpage imposé : les n_train premiers cycles / le reste (test final)."""
    return X.iloc[:n_train], X.iloc[n_train:], y.iloc[:n_train], y.iloc[n_train:]
