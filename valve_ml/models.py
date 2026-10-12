"""Définition des modèles candidats sous forme de Pipelines homogènes.

Chaque pipeline enchaîne : filtre de corrélation -> standardisation ->
sélection univariée (ANOVA F) -> classifieur. Les arbres et ensembles n'ont
pas besoin de standardisation mais elle ne les pénalise pas ; on garde ainsi
exactement le même prétraitement pour tous les modèles.
"""
from __future__ import annotations

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .selection import CorrelationFilter

RANDOM_STATE = 42


def make_pipeline(clf, k="all", corr_threshold: float = 0.95) -> Pipeline:
    return Pipeline(
        [
            ("corr", CorrelationFilter(threshold=corr_threshold)),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=k)),
            ("clf", clf),
        ]
    )


def base_classifiers() -> dict:
    """Classifieurs « par défaut » (avant optimisation des hyperparamètres)."""
    return {
        "Régression logistique": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "Arbre de décision": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Naive Bayes": GaussianNB(),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
    }


def stacking_classifier() -> StackingClassifier:
    """Ensemble hétérogène : méta-modèle logistique sur LR, SVM et RF."""
    return StackingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
            ("svm", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
            ("rf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE)),
        ],
        final_estimator=LogisticRegression(max_iter=5000),
        cv=5,
        n_jobs=-1,
    )


def all_pipelines(k="all") -> dict:
    pipes = {name: make_pipeline(clf, k=k) for name, clf in base_classifiers().items()}
    pipes["Stacking (LR+SVM+RF)"] = make_pipeline(stacking_classifier(), k=k)
    return pipes
