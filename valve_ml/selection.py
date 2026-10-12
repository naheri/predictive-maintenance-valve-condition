"""Sélection de variables utilisable à l'intérieur d'un Pipeline scikit-learn.

La sélection est apprise sur les plis d'entraînement uniquement, ce qui évite
toute fuite d'information vers les plis de validation ou le jeu de test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Supprime une variable de chaque paire dont |corrélation| > threshold.

    Parcourt les colonnes dans l'ordre et garde une colonne seulement si elle
    n'est pas trop corrélée avec une colonne déjà gardée.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        Xa = np.asarray(X, dtype=float)
        std = Xa.std(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.nan_to_num(np.abs(np.corrcoef(Xa, rowvar=False)))
        keep: list[int] = []
        for j in range(Xa.shape[1]):
            if std[j] == 0:
                continue  # variable constante
            if all(corr[j, k] <= self.threshold for k in keep):
                keep.append(j)
        self.support_ = np.zeros(Xa.shape[1], dtype=bool)
        self.support_[keep] = True
        self.feature_names_in_ = (
            np.asarray(X.columns) if isinstance(X, pd.DataFrame) else None
        )
        self.n_features_in_ = Xa.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "support_")
        if isinstance(X, pd.DataFrame):
            return X.loc[:, self.support_]
        return np.asarray(X)[:, self.support_]

    def get_support(self):
        check_is_fitted(self, "support_")
        return self.support_

    def get_feature_names_out(self, input_features=None):
        names = input_features if input_features is not None else self.feature_names_in_
        if names is None:
            names = np.array([f"x{i}" for i in range(self.n_features_in_)])
        return np.asarray(names)[self.support_]
