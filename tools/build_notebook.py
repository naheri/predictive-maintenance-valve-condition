"""Génère notebooks/projet_valve.ipynb.

Le notebook est autonome (il peut être ouvert seul dans Jupyter ou Colab) : le
code des modules valve_ml/data.py, features.py et selection.py y est recopié
tel quel pour qu'il n'y ait qu'une seule source de vérité.
Usage : python tools/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def module_source(name: str) -> str:
    src = (ROOT / "valve_ml" / name).read_text(encoding="utf-8")
    return src.replace("from __future__ import annotations\n", "")


cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


# ---------------------------------------------------------------------------
md(r"""
# Projet de Machine Learning — Maintenance prédictive d'un banc hydraulique

**Objectif.** Prédire, pour chaque cycle de production de 60 s, si la condition de la
valve est **optimale (= 100 %)** ou non, à partir de deux capteurs :

| Capteur | Grandeur | Unité | Échantillonnage | Points / cycle |
|---|---|---|---|---|
| PS2 | Pression | bar | 100 Hz | 6000 |
| FS1 | Débit volumique | l/min | 10 Hz | 600 |

et du fichier `profile.txt` (conditions des composants). Les **2000 premiers cycles** servent à
construire le modèle, les **205 derniers** constituent l'échantillon de test final, qui n'est
utilisé qu'une seule fois, à la toute fin.

Plan :
1. Prétraitement des données (chargement, qualité, cible binaire, visualisations, extraction
   et sélection de variables)
2. Modélisation (choix et justification des modèles)
3. Évaluation (validation croisée, matrices de confusion, ROC/AUC, biais/variance)
4. Optimisation des hyperparamètres (Grid Search, Random Search)
5. Test final et interprétation
""")

code(r"""
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier)
from sklearn.feature_selection import (SelectKBest, VarianceThreshold, f_classif,
                                       mutual_info_classif)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_predict,
                                     cross_validate, learning_curve)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 50, "display.width", 140)

# Dossier où sont enregistrées les figures et tables utilisées par le rapport LaTeX
REPORT_DIR = Path("../rapport") if Path("../rapport").exists() else Path("rapport")
FIG_DIR, TAB_DIR = REPORT_DIR / "figures", REPORT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True); TAB_DIR.mkdir(parents=True, exist_ok=True)

def savefig(name):
    plt.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")

RESULTS = {}  # chiffres clés exportés vers le rapport (rapport/tables/resultats.tex)
""")

# ---------------------------------------------------------------------------
md(r"""
## 1. Prétraitement des données

### 1.1 Chargement

Les fonctions ci-dessous cherchent les fichiers `PS2.txt`, `FS1.txt` et `profile.txt` dans
`./data`, `../data`, `../Projet_ML`… ou dans le dossier indiqué par la variable
d'environnement `VALVE_DATA_DIR`. Sous Colab, déposez les trois fichiers dans `/content/data`.
""")
code(module_source("data.py"))
code(r"""
DATA_DIR = find_data_dir()
print("Données lues dans :", DATA_DIR)
t0 = time.time()
ps2, fs1, profile = load_all(DATA_DIR)
print(f"Chargement en {time.time()-t0:.1f} s")
print("PS2     :", ps2.shape, "(cycles x points à 100 Hz)")
print("FS1     :", fs1.shape, "(cycles x points à 10 Hz)")
print("profile :", profile.shape)
profile.head()
""")

md("### 1.2 Statistiques descriptives")
code(r"""
def describe_signal(x, name, unit):
    flat = x.ravel()
    per_cycle_mean = x.mean(axis=1)
    return pd.Series({
        "capteur": name, "unité": unit, "cycles": x.shape[0], "points/cycle": x.shape[1],
        "min": flat.min(), "max": flat.max(), "moyenne": flat.mean(), "écart-type": flat.std(),
        "médiane": np.median(flat),
        "écart-type des moyennes par cycle": per_cycle_mean.std(),
    })

desc = pd.DataFrame([describe_signal(ps2, "PS2", "bar"), describe_signal(fs1, "FS1", "l/min")]).set_index("capteur")
desc
""")
code(r"""
profile.describe().T
""")
code(r"""
for col in profile.columns:
    print(col, profile[col].value_counts().sort_index().to_dict())
""")

md(r"""
### 1.3 Valeurs manquantes et valeurs aberrantes

On vérifie : valeurs manquantes (`NaN`), valeurs infinies, valeurs physiquement impossibles
(pression ou débit négatifs), cycles constants (capteur figé), puis on recherche des cycles
atypiques avec la règle de l'IQR (1,5 × écart interquartile) et le z-score robuste sur des
statistiques par cycle.
""")
code(r"""
quality = pd.DataFrame({
    "NaN": [np.isnan(ps2).sum(), np.isnan(fs1).sum(), profile.isna().sum().sum()],
    "inf": [np.isinf(ps2).sum(), np.isinf(fs1).sum(), 0],
    "valeurs < 0": [(ps2 < 0).sum(), (fs1 < 0).sum(), (profile < 0).sum().sum()],
    "cycles constants": [(ps2.std(axis=1) == 0).sum(), (fs1.std(axis=1) == 0).sum(), np.nan],
    "lignes dupliquées": [pd.DataFrame(ps2).duplicated().sum(), pd.DataFrame(fs1).duplicated().sum(), np.nan],
}, index=["PS2", "FS1", "profile"])
quality
""")
code(r"""
def iqr_outliers(v):
    q1, q3 = np.percentile(v, [25, 75]); iqr = q3 - q1
    return (v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)

def robust_z(v):
    med = np.median(v); mad = np.median(np.abs(v - med)) * 1.4826
    return (v - med) / (mad + 1e-12)

cycle_stats = pd.DataFrame({
    "PS2_mean": ps2.mean(axis=1), "PS2_std": ps2.std(axis=1), "PS2_max": ps2.max(axis=1),
    "FS1_mean": fs1.mean(axis=1), "FS1_std": fs1.std(axis=1), "FS1_max": fs1.max(axis=1),
}, index=profile.index)
out_iqr = cycle_stats.apply(iqr_outliers)
out_z = cycle_stats.apply(lambda c: np.abs(robust_z(c.values)) > 3.5)
outlier_summary = pd.DataFrame({"IQR (1,5)": out_iqr.sum(), "|z robuste| > 3,5": out_z.sum()})
display(outlier_summary)
any_out = out_iqr.any(axis=1)
print(f"Cycles signalés par au moins un critère IQR : {any_out.sum()} ({any_out.mean():.1%})")
print("Répartition de ces cycles selon la condition de la valve :")
print(profile.loc[any_out, "valve"].value_counts().sort_index().to_dict())
print("Répartition selon la condition du refroidisseur :")
print(profile.loc[any_out, "cooler"].value_counts().sort_index().to_dict())
RESULTS["nOutlierCycles"] = int(any_out.sum())
""")
code(r"""
fig, axes = plt.subplots(1, 6, figsize=(16, 3.6))
for ax, col in zip(axes, cycle_stats.columns):
    sns.boxplot(y=cycle_stats[col], ax=ax, color="#8fb3d9", fliersize=2)
    ax.set_title(col); ax.set_ylabel("")
plt.suptitle("Statistiques par cycle : détection des valeurs aberrantes", y=1.03)
plt.tight_layout(); savefig("boxplots_cycles_bruts"); plt.show()
""")
md(r"""
**Décision.** Les cycles « aberrants » au sens statistique ne sont pas des erreurs de mesure :
ils correspondent à des états physiques réels du banc (refroidisseur proche de la panne,
accumulateur dégradé, régime transitoire…). Les supprimer ferait disparaître justement les
situations que la maintenance prédictive doit reconnaître, et le jeu de test contiendra
des cycles de ce type. On les **conserve** et l'on limite leur influence par la
standardisation et par le choix de modèles robustes (arbres, ensembles). Aucune valeur
manquante n'est à imputer.
""")

md(r"""
### 1.4 Encodage des variables catégorielles

Les capteurs fournissent uniquement des variables **continues**. Les colonnes de
`profile.txt` sont des niveaux numériques ordonnés (variables ordinales) ; elles ne servent
pas de variables explicatives (ce sont des étiquettes d'état mesurées hors ligne, inconnues
en production), sauf `valve` qui définit la cible. Aucun encodage one-hot n'est donc
nécessaire ; la seule transformation est la **binarisation de la cible**.
""")

md("### 1.5 Transformation en classification binaire et équilibre des classes")
code(r"""
y = make_target(profile)
y_train, y_test = y.iloc[:N_TRAIN], y.iloc[N_TRAIN:]

dist = pd.DataFrame({
    "Total": y.value_counts(), "Entraînement (1-2000)": y_train.value_counts(),
    "Test (2001-2205)": y_test.value_counts(),
}).rename(index={1: "optimale (1)", 0: "non optimale (0)"})
dist_pct = (dist / dist.sum() * 100).round(1)
display(dist); display(dist_pct)
RESULTS.update(trainPosPct=dist_pct.iloc[:, 1]["optimale (1)"], testPosPct=dist_pct.iloc[:, 2]["optimale (1)"],
               nTest=len(y_test), nTestPos=int(y_test.sum()))

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
profile["valve"].value_counts().sort_index().plot.bar(ax=axes[0], color="#5b8bd0")
axes[0].set_title("Condition de la valve (%)"); axes[0].set_xlabel("valve"); axes[0].set_ylabel("cycles")
for ax, (name, s) in zip(axes[1:], [("Entraînement", y_train), ("Test", y_test)]):
    s.map({1: "optimale", 0: "non optimale"}).value_counts().reindex(["optimale", "non optimale"]).plot.bar(ax=ax, color=["#3a9d5d", "#d0605e"])
    ax.set_title(f"{name} (n={len(s)})"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=0)
plt.tight_layout(); savefig("distribution_classes"); plt.show()
""")
md(r"""
Le jeu d'entraînement est **quasiment équilibré** (≈ 53 % / 47 %) : il n'est pas nécessaire de
rééchantillonner. On utilisera tout de même une validation croisée **stratifiée** et on
suivra le F1-score et l'AUC en plus de l'accuracy. Le jeu de test est en revanche
déséquilibré dans l'autre sens (≈ 36 % de cycles optimaux) : l'accuracy y sera moins
informative que la précision, le rappel et l'AUC.

Il faut aussi noter que le découpage est **temporel** (cycles 1-2000 / 2001-2205) et que
les conditions des autres composants changent au fil de l'expérience :
""")
code(r"""
comp = pd.concat({"Entraînement": profile.iloc[:N_TRAIN], "Test": profile.iloc[N_TRAIN:]}, names=["jeu"])
for col in ["cooler", "pump", "accumulator", "stable"]:
    display(pd.crosstab(comp.index.get_level_values(0), comp[col], normalize="index").round(3).rename_axis(index="jeu", columns=col))
""")
code(r"""
fig, ax = plt.subplots(figsize=(14, 3.5))
for col, c in zip(["cooler", "valve", "pump", "accumulator"], ["#5b8bd0", "#d0605e", "#3a9d5d", "#9b6fc0"]):
    v = profile[col]; ax.plot(profile.index, (v - v.min()) / (v.max() - v.min()), label=col, lw=1, color=c)
ax.axvline(N_TRAIN + 0.5, color="k", ls="--"); ax.text(N_TRAIN + 10, 1.02, "test", fontsize=10)
ax.set_xlabel("cycle"); ax.set_ylabel("niveau normalisé"); ax.legend(ncol=4, loc="lower left")
ax.set_title("Évolution des conditions des composants au cours de l'expérience")
plt.tight_layout(); savefig("conditions_temporelles"); plt.show()
""")

md(r"""
### 1.6 Visualisation des signaux bruts

On compare le profil moyen d'un cycle selon la condition de la valve. Un retard de
commutation doit se traduire par un **décalage temporel** et une modification des
transitoires de pression et de débit.
""")
code(r"""
t_ps2 = np.arange(ps2.shape[1]) / 100.0
t_fs1 = np.arange(fs1.shape[1]) / 10.0
palette = {100: "#3a9d5d", 90: "#e0a33a", 80: "#d0605e", 73: "#7a2a2a"}
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
for lvl in [100, 90, 80, 73]:
    m = (profile["valve"] == lvl).values
    axes[0].plot(t_ps2, ps2[m].mean(axis=0), color=palette[lvl], lw=1.2, label=f"valve {lvl} %")
    axes[1].plot(t_fs1, fs1[m].mean(axis=0), color=palette[lvl], lw=1.2, label=f"valve {lvl} %")
axes[0].set_ylabel("PS2 moyen (bar)"); axes[1].set_ylabel("FS1 moyen (l/min)"); axes[1].set_xlabel("temps dans le cycle (s)")
axes[0].legend(ncol=4); axes[0].set_title("Profil moyen d'un cycle selon la condition de la valve")
plt.tight_layout(); savefig("signaux_moyens_par_valve"); plt.show()
""")
code(r"""
fig, axes = plt.subplots(2, 2, figsize=(14, 6.5))
for j, (zoom, title) in enumerate([((0, 60), "cycle complet"), ((0, 6), "zoom 0-6 s")]):
    for lvl in [100, 73]:
        idx = np.where(profile["valve"] == lvl)[0][:3]
        for i in idx:
            axes[0, j].plot(t_ps2, ps2[i], color=palette[lvl], lw=0.7, alpha=0.8)
            axes[1, j].plot(t_fs1, fs1[i], color=palette[lvl], lw=0.7, alpha=0.8)
    for a in axes[:, j]: a.set_xlim(*zoom)
    axes[0, j].set_title(f"Exemples de cycles ({title}) : vert = 100 %, rouge foncé = 73 %")
axes[0, 0].set_ylabel("PS2 (bar)"); axes[1, 0].set_ylabel("FS1 (l/min)")
axes[1, 0].set_xlabel("temps (s)"); axes[1, 1].set_xlabel("temps (s)")
plt.tight_layout(); savefig("exemples_cycles"); plt.show()
""")

md(r"""
### 1.7 Extraction de caractéristiques

Les 6600 points bruts d'un cycle sont très redondants (points voisins quasi identiques) et
peu adaptés à des modèles comme k-NN, Naive Bayes ou un arbre unique. On résume chaque
signal par des **descripteurs physiques interprétables** :

* statistiques globales : moyenne, écart-type, min, max, quantiles, IQR, amplitude, RMS,
  asymétrie, aplatissement, instants du maximum et du minimum ;
* statistiques de la **dérivée** (vitesse de variation, raideur des transitoires) ;
* moyennes et écarts-types sur **10 fenêtres temporelles** de 6 s (le retard de la valve
  décale les phases du cycle) ;
* répartition de l'**énergie spectrale** dans 4 bandes de fréquence.
""")
code(module_source("features.py"))
code(r"""
t0 = time.time()
X = build_feature_matrix(ps2, fs1)
print(f"Matrice de caractéristiques : {X.shape} en {time.time()-t0:.1f} s")
print("NaN :", X.isna().sum().sum(), "| inf :", np.isinf(X.values).sum())
X_train, X_test = X.iloc[:N_TRAIN], X.iloc[N_TRAIN:]
RESULTS["nFeatures"] = X.shape[1]
X.head()
""")

md(r"""
### 1.8 Analyse des corrélations

Toutes les analyses de cette section sont faites **sur le jeu d'entraînement uniquement**.
""")
code(r"""
corr_target = X_train.apply(lambda c: c.corr(y_train)).sort_values(key=np.abs, ascending=False)
top_corr = corr_target.head(15)
fig, ax = plt.subplots(figsize=(9, 5.5))
top_corr.iloc[::-1].plot.barh(ax=ax, color=np.where(top_corr.iloc[::-1] > 0, "#3a9d5d", "#d0605e"))
ax.set_xlabel("corrélation de Pearson avec la cible (valve optimale)")
ax.set_title("15 variables les plus corrélées à la cible")
plt.tight_layout(); savefig("correlation_cible"); plt.show()
top_corr.to_frame("corr").round(3)
""")
code(r"""
top_feats = corr_target.head(20).index
cm = X_train[top_feats].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .7})
plt.title("Matrice de corrélation des 20 variables les plus liées à la cible")
plt.tight_layout(); savefig("matrice_correlation"); plt.show()

with np.errstate(invalid="ignore", divide="ignore"):
    full_corr = np.nan_to_num(X_train.corr().abs().values)
iu = np.triu_indices_from(full_corr, k=1)
frac_high = np.mean(full_corr[iu] > 0.95)
print(f"Part des paires de variables avec |r| > 0,95 : {frac_high:.1%}")
RESULTS["pctHighCorrPairs"] = round(100 * frac_high, 1)
""")
code(r"""
plot_feats = list(corr_target.head(6).index)
fig, axes = plt.subplots(2, 6, figsize=(18, 6.5))
lab = y_train.map({1: "optimale", 0: "non optimale"})
for j, f in enumerate(plot_feats):
    sns.histplot(x=X_train[f], hue=lab, ax=axes[0, j], bins=40, stat="density", common_norm=False,
                 palette={"optimale": "#3a9d5d", "non optimale": "#d0605e"}, legend=(j == 0))
    sns.boxplot(x=lab, y=X_train[f], ax=axes[1, j], hue=lab, palette={"optimale": "#3a9d5d", "non optimale": "#d0605e"}, fliersize=2, legend=False)
    axes[0, j].set_title(f, fontsize=9); axes[0, j].set_xlabel(""); axes[1, j].set_xlabel(""); axes[1, j].set_ylabel("")
plt.suptitle("Histogrammes et boxplots des variables les plus discriminantes", y=1.01)
plt.tight_layout(); savefig("histogrammes_boxplots"); plt.show()
""")
code(r"""
# Relation entre la valve et les autres conditions (analyse des causes)
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
for ax, col in zip(axes, ["cooler", "pump", "accumulator", "stable"]):
    ct = pd.crosstab(profile[col], profile["valve"] == 100, normalize="index")[True]
    ct.plot.bar(ax=ax, color="#5b8bd0"); ax.set_ylim(0, 1); ax.set_title(f"P(valve optimale | {col})")
    ax.set_ylabel("proportion"); ax.tick_params(axis="x", rotation=0)
plt.tight_layout(); savefig("valve_vs_autres_conditions"); plt.show()
cramers = {}
for col in ["cooler", "pump", "accumulator", "stable"]:
    ct = pd.crosstab(profile[col], profile["valve"])
    chi2 = stats.chi2_contingency(ct)[0]
    cramers[col] = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
pd.Series(cramers, name="V de Cramér avec valve").round(3).to_frame()
""")
md(r"""
**Lecture.** L'état du refroidisseur est indépendant de celui de la valve (V ≈ 0). Les
associations avec la pompe, l'accumulateur et surtout l'indicateur de stabilité viennent du
**plan d'expérience** et non d'un lien physique : chaque niveau dégradé de la valve (73, 80,
90 %) est testé exactement 120 fois pour chaque état de la pompe, et tous les cycles « non
stables » (phases de transition après un changement de configuration) ont été enregistrés avec
une valve à 100 %. Ces variables ne sont donc pas des causes de la dégradation de la valve
(et ne sont de toute façon pas disponibles en production) ; la dégradation doit être détectée
dans la **signature des signaux** de pression et de débit.
""")

md(r"""
### 1.9 Sélection de variables

On combine plusieurs familles de méthodes :

1. **Filtre de variance** : suppression des variables (quasi) constantes ;
2. **Filtre de corrélation** : pour chaque paire avec |r| > 0,95, une seule variable est
   conservée (réduction de la redondance, utile pour Naive Bayes et la régression logistique) ;
3. **Méthodes univariées** : test ANOVA F et information mutuelle ;
4. **Méthode embarquée** : régression logistique pénalisée L1 et importance des variables
   d'une forêt aléatoire.

Dans les pipelines de modélisation, le filtre de corrélation et la sélection `SelectKBest`
(ANOVA F) sont placés **à l'intérieur du pipeline** : ils sont réappris dans chaque pli de
validation croisée, ce qui évite toute fuite d'information.
""")
code(module_source("selection.py"))
code(r"""
vt = VarianceThreshold(1e-8).fit(X_train)
X_tr_v = X_train.loc[:, vt.get_support()]
cf = CorrelationFilter(0.95).fit(X_tr_v)
X_tr_c = cf.transform(X_tr_v)
print(f"Variables initiales             : {X_train.shape[1]}")
print(f"Après filtre de variance        : {X_tr_v.shape[1]}")
print(f"Après filtre de corrélation 0,95: {X_tr_c.shape[1]}")
RESULTS["nAfterCorr"] = X_tr_c.shape[1]

Xs = StandardScaler().fit_transform(X_tr_c)
F, pval = f_classif(Xs, y_train)
mi = mutual_info_classif(Xs, y_train, random_state=RANDOM_STATE)
l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=5000).fit(Xs, y_train)
rf_sel = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=RANDOM_STATE).fit(X_tr_c, y_train)
scores = pd.DataFrame({
    "ANOVA F": F, "p-value": pval, "info. mutuelle": mi,
    "|coef L1|": np.abs(l1.coef_[0]), "importance RF": rf_sel.feature_importances_,
}, index=X_tr_c.columns)
ranks = scores.drop(columns="p-value").rank(ascending=False)
scores["rang moyen"] = ranks.mean(axis=1)
scores = scores.sort_values("rang moyen")
print(f"Variables non nulles avec L1 (C=0,1) : {(scores['|coef L1|'] > 0).sum()}")
RESULTS["nL1NonZero"] = int((scores["|coef L1|"] > 0).sum())
scores.head(15).round(4)
""")
code(r"""
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, col, c in zip(axes, ["ANOVA F", "info. mutuelle", "|coef L1|", "importance RF"],
                      ["#5b8bd0", "#9b6fc0", "#e0a33a", "#3a9d5d"]):
    s = scores[col].sort_values(ascending=False).head(12).iloc[::-1]
    s.plot.barh(ax=ax, color=c); ax.set_title(col); ax.tick_params(axis="y", labelsize=8)
plt.suptitle("Classement des variables selon quatre méthodes de sélection", y=1.02)
plt.tight_layout(); savefig("selection_variables"); plt.show()
""")
md(r"""
**Choix du nombre de variables `k`.** On mesure en validation croisée l'AUC d'une régression
logistique et d'une forêt aléatoire en fonction du nombre de variables retenues par ANOVA F.
""")
code(r"""
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
n_max = X_tr_c.shape[1]
k_grid = sorted({k for k in [2, 5, 10, 15, 20, 30, 40, 60, n_max] if k <= n_max})

def sel_pipe(clf, k):
    return Pipeline([("corr", CorrelationFilter(0.95)), ("scaler", StandardScaler()),
                     ("select", SelectKBest(f_classif, k=k)), ("clf", clf)])

curve = []
for k in k_grid:
    for name, clf in [("Régression logistique", LogisticRegression(max_iter=5000)),
                      ("Random Forest", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE))]:
        s = cross_validate(sel_pipe(clf, k), X_train, y_train, cv=CV, scoring=["roc_auc", "f1"])
        curve.append({"k": k, "modèle": name, "AUC": s["test_roc_auc"].mean(), "F1": s["test_f1"].mean()})
curve = pd.DataFrame(curve)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, m in zip(axes, ["AUC", "F1"]):
    sns.lineplot(data=curve, x="k", y=m, hue="modèle", marker="o", ax=ax)
    ax.set_xscale("log"); ax.set_title(f"{m} (CV 5 plis) selon le nombre de variables k")
plt.tight_layout(); savefig("choix_k"); plt.show()
curve.pivot(index="k", columns="modèle", values=["AUC", "F1"]).round(4)
""")
code(r"""
# k par défaut : plus petit k dont l'AUC moyenne (2 modèles) est à moins de 0,005 du maximum
mean_auc = curve.groupby("k")["AUC"].mean()
K_DEFAULT = int(mean_auc[mean_auc >= mean_auc.max() - 0.005].index.min())
print("k retenu par défaut :", K_DEFAULT, "(k sera de nouveau optimisé à l'étape 4)")
RESULTS["kDefault"] = K_DEFAULT
""")

# ---------------------------------------------------------------------------
md(r"""
## 2. Modélisation

| Modèle | Nature | Frontière de décision | Hypothèses / adaptation aux données |
|---|---|---|---|
| Régression logistique | paramétrique, linéaire | hyperplan | rapide, interprétable, sensible à la colinéarité (d'où le filtre de corrélation) ; référence linéaire |
| k-NN | non paramétrique, à base d'instances | non linéaire, locale | nécessite des variables standardisées ; souffre du bruit et de la grande dimension |
| Arbre de décision | non paramétrique | non linéaire, par morceaux | insensible aux échelles et aux valeurs extrêmes, interprétable ; forte variance |
| Naive Bayes gaussien | probabiliste génératif | quadratique | suppose des variables indépendantes et gaussiennes : hypothèse violée ici (variables corrélées), sert de référence simple |
| SVM (noyau RBF) | à marge maximale | non linéaire (noyau) | performant en dimension moyenne, robuste au bruit grâce à la marge ; sensible à C et γ |
| Random Forest | ensemble (bagging) | non linéaire | réduit la variance des arbres, robuste au bruit et aux valeurs extrêmes |
| Gradient Boosting / AdaBoost | ensemble (boosting) | non linéaire | réduit le biais en corrigeant séquentiellement les erreurs ; risque de sur-apprentissage si mal régularisé |
| Stacking (LR + SVM + RF) | ensemble hétérogène | combinaison | combine des modèles de biais différents via un méta-modèle logistique |

Tous les modèles partagent le même pipeline :
`CorrelationFilter(0,95) → StandardScaler → SelectKBest(ANOVA F, k) → classifieur`.
""")
code(r"""
def make_pipeline(clf, k=None):
    return sel_pipe(clf, K_DEFAULT if k is None else k)

base_models = {
    "Régression logistique": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "Arbre de décision": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Naive Bayes": GaussianNB(),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
    "Stacking (LR+SVM+RF)": StackingClassifier(
        estimators=[("lr", LogisticRegression(max_iter=5000)),
                    ("svm", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
                    ("rf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE))],
        final_estimator=LogisticRegression(max_iter=5000), cv=5, n_jobs=-1),
}
pipelines = {name: make_pipeline(clf) for name, clf in base_models.items()}
""")

# ---------------------------------------------------------------------------
md(r"""
## 3. Évaluation des modèles

### 3.1 Validation croisée stratifiée (5 plis) sur les 2000 cycles d'entraînement

Pour chaque modèle on mesure accuracy, précision, rappel, F1 et AUC sur les plis de
validation, ainsi que les scores sur les plis d'entraînement : l'écart entre les deux
renseigne sur la **variance** (sur-apprentissage), le niveau du score de validation sur le
**biais**.
""")
code(r"""
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

def cv_table(pipes, cv=CV):
    rows = []
    for name, pipe in pipes.items():
        t0 = time.time()
        s = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, return_train_score=True, n_jobs=-1)
        row = {"modèle": name}
        for m in SCORING:
            row[m] = s[f"test_{m}"].mean(); row[f"{m}_std"] = s[f"test_{m}"].std()
        row["F1 train"] = s["train_f1"].mean()
        row["écart train-val (F1)"] = row["F1 train"] - row["f1"]
        row["temps (s)"] = time.time() - t0
        rows.append(row)
    return pd.DataFrame(rows).set_index("modèle")

res_base = cv_table(pipelines).sort_values("roc_auc", ascending=False)
res_base[["accuracy", "precision", "recall", "f1", "roc_auc", "roc_auc_std", "F1 train", "écart train-val (F1)", "temps (s)"]].round(4)
""")
code(r"""
fig, ax = plt.subplots(figsize=(12, 4.5))
plot_df = res_base[["accuracy", "precision", "recall", "f1", "roc_auc"]].rename(columns={"roc_auc": "AUC"})
plot_df.plot.bar(ax=ax, width=0.8, colormap="tab10")
lo = max(0.0, plot_df.values.min() - 0.05)
ax.set_ylim(lo, 1.0); ax.set_ylabel("score moyen (CV 5 plis)"); ax.tick_params(axis="x", rotation=25)
ax.legend(ncol=5, loc="lower right"); ax.set_title("Comparaison des modèles avant optimisation")
plt.tight_layout(); savefig("comparaison_modeles_base"); plt.show()
""")

md(r"""
### 3.2 Matrices de confusion et courbes ROC (prédictions hors-pli)

Les prédictions sont obtenues par `cross_val_predict` : chaque cycle d'entraînement est
prédit par un modèle qui ne l'a pas vu.
""")
code(r"""
oof_proba = {}
for name, pipe in pipelines.items():
    oof_proba[name] = cross_val_predict(pipe, X_train, y_train, cv=CV, method="predict_proba", n_jobs=-1)[:, 1]

names = list(pipelines)
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for ax, name in zip(axes.ravel(), names):
    ConfusionMatrixDisplay.from_predictions(y_train, (oof_proba[name] >= 0.5).astype(int), ax=ax, colorbar=False,
                                            display_labels=["non opt.", "optimale"], cmap="Blues")
    ax.set_title(name, fontsize=10)
axes.ravel()[-1].axis("off")
plt.suptitle("Matrices de confusion (validation croisée, seuil 0,5)", y=1.0)
plt.tight_layout(); savefig("matrices_confusion_cv"); plt.show()
""")
code(r"""
fig, ax = plt.subplots(figsize=(7.5, 6.5))
for name in names:
    fpr, tpr, _ = roc_curve(y_train, oof_proba[name])
    ax.plot(fpr, tpr, lw=1.4, label=f"{name} (AUC={roc_auc_score(y_train, oof_proba[name]):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("taux de faux positifs"); ax.set_ylabel("taux de vrais positifs")
ax.set_title("Courbes ROC (validation croisée)"); ax.legend(fontsize=8, loc="lower right")
plt.tight_layout(); savefig("roc_cv"); plt.show()
""")

md(r"""
### 3.3 Compromis biais / variance : courbes d'apprentissage

Une courbe d'entraînement très haute avec un écart important à la courbe de validation
traduit une **variance** élevée ; deux courbes basses et proches traduisent un **biais**
élevé.
""")
code(r"""
lc_models = ["Régression logistique", "Naive Bayes", "Arbre de décision", "k-NN", "SVM (RBF)", "Random Forest"]
fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
for ax, name in zip(axes.ravel(), lc_models):
    sizes, tr, va = learning_curve(pipelines[name], X_train, y_train, cv=CV, scoring="f1",
                                   train_sizes=np.linspace(0.1, 1.0, 8), shuffle=True,
                                   random_state=RANDOM_STATE, n_jobs=-1)
    ax.plot(sizes, tr.mean(1), "o-", color="#d0605e", label="entraînement")
    ax.fill_between(sizes, tr.mean(1) - tr.std(1), tr.mean(1) + tr.std(1), color="#d0605e", alpha=.15)
    ax.plot(sizes, va.mean(1), "o-", color="#3a9d5d", label="validation")
    ax.fill_between(sizes, va.mean(1) - va.std(1), va.mean(1) + va.std(1), color="#3a9d5d", alpha=.15)
    ax.set_title(name); ax.set_xlabel("taille du jeu d'entraînement")
axes[0, 0].set_ylabel("F1"); axes[1, 0].set_ylabel("F1"); axes[0, 0].legend(loc="lower right")
plt.suptitle("Courbes d'apprentissage", y=1.0)
plt.tight_layout(); savefig("courbes_apprentissage"); plt.show()
""")

md(r"""
### 3.4 Robustesse de la validation : plis contigus vs plis mélangés

Les cycles successifs se ressemblent (mêmes conditions maintenues pendant plusieurs
cycles). Une validation croisée à plis **mélangés** peut donc être optimiste. On la compare
à une validation à **plis contigus** (`KFold` sans mélange), plus proche de la situation du
test final (cycles futurs).
""")
code(r"""
CV_BLOCK = KFold(n_splits=5, shuffle=False)
rob = []
for name in ["Régression logistique", "SVM (RBF)", "Random Forest", "Gradient Boosting"]:
    a = cross_validate(pipelines[name], X_train, y_train, cv=CV, scoring=["f1", "roc_auc"], n_jobs=-1)
    b = cross_validate(pipelines[name], X_train, y_train, cv=CV_BLOCK, scoring=["f1", "roc_auc"], n_jobs=-1)
    rob.append({"modèle": name, "F1 plis mélangés": a["test_f1"].mean(), "F1 plis contigus": b["test_f1"].mean(),
                "AUC plis mélangés": a["test_roc_auc"].mean(), "AUC plis contigus": b["test_roc_auc"].mean()})
rob = pd.DataFrame(rob).set_index("modèle"); rob.round(4)
""")

# ---------------------------------------------------------------------------
md(r"""
## 4. Optimisation des hyperparamètres

* **Grid Search** (recherche exhaustive) pour les modèles qui ont peu d'hyperparamètres :
  régression logistique, k-NN, arbre de décision, SVM, Naive Bayes ;
* **Random Search** (tirage aléatoire de 40 combinaisons) pour les ensembles dont l'espace
  est plus grand : Random Forest et Gradient Boosting.

Le nombre de variables `k` de `SelectKBest` est optimisé **en même temps** que les
hyperparamètres du classifieur. Le critère est l'AUC en validation croisée stratifiée à 5
plis ; chaque recherche est de plus réévaluée en F1.
""")
code(r"""
k_opts = sorted({k for k in [5, 10, 20, 30, 40, n_max] if k <= n_max})
searches = {
    "Régression logistique": ("grid", {"select__k": k_opts, "clf__C": [0.01, 0.1, 1, 10, 100],
                                       "clf__penalty": ["l1", "l2"], "clf__solver": ["liblinear"]}),
    "k-NN": ("grid", {"select__k": k_opts, "clf__n_neighbors": [3, 5, 7, 11, 15, 21, 31],
                      "clf__weights": ["uniform", "distance"], "clf__p": [1, 2]}),
    "Arbre de décision": ("grid", {"select__k": k_opts, "clf__max_depth": [3, 5, 8, 12, None],
                                   "clf__min_samples_leaf": [1, 5, 10, 20], "clf__criterion": ["gini", "entropy"]}),
    "Naive Bayes": ("grid", {"select__k": k_opts, "clf__var_smoothing": np.logspace(-11, -3, 9)}),
    "SVM (RBF)": ("grid", {"select__k": k_opts, "clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.001, 0.01, 0.1]}),
    "Random Forest": ("random", {"select__k": k_opts, "clf__n_estimators": randint(100, 600),
                                 "clf__max_depth": [None, 5, 10, 20], "clf__min_samples_leaf": randint(1, 10),
                                 "clf__max_features": ["sqrt", "log2", 0.3, 0.5]}),
    "Gradient Boosting": ("random", {"select__k": k_opts, "clf__n_estimators": randint(100, 500),
                                     "clf__learning_rate": loguniform(0.01, 0.3), "clf__max_depth": randint(2, 6),
                                     "clf__subsample": uniform(0.6, 0.4)}),
}

best = {}; tuning_rows = []
for name, (kind, space) in searches.items():
    t0 = time.time()
    if kind == "grid":
        search = GridSearchCV(pipelines[name], space, cv=CV, scoring="roc_auc", n_jobs=-1, refit=True)
    else:
        search = RandomizedSearchCV(pipelines[name], space, n_iter=40, cv=CV, scoring="roc_auc",
                                    n_jobs=-1, refit=True, random_state=RANDOM_STATE)
    search.fit(X_train, y_train)
    best[name] = search.best_estimator_
    n_cand = len(search.cv_results_["params"])
    tuning_rows.append({"modèle": name, "méthode": "Grid" if kind == "grid" else "Random",
                        "combinaisons": n_cand, "AUC avant": res_base.loc[name, "roc_auc"],
                        "AUC après": search.best_score_, "temps (s)": time.time() - t0,
                        "meilleurs paramètres": {k.replace("clf__", "").replace("select__", ""): (float(f"{v:.4g}") if isinstance(v, float) else v)
                                                 for k, v in search.best_params_.items()}})
    print(f"{name:22s} {kind:6s} AUC={search.best_score_:.4f}  ({time.time()-t0:.0f} s)  {search.best_params_}")
tuning = pd.DataFrame(tuning_rows).set_index("modèle")
tuning[["méthode", "combinaisons", "AUC avant", "AUC après", "temps (s)"]].round(4)
""")
code(r"""
# Ensemble final : stacking des trois meilleurs modèles optimisés
top3 = tuning.sort_values("AUC après", ascending=False).index[:3].tolist()
print("Modèles empilés :", top3)
stack_opt = StackingClassifier(estimators=[(f"m{i}", clone(best[n])) for i, n in enumerate(top3)],
                               final_estimator=LogisticRegression(max_iter=5000), cv=5, n_jobs=-1)
best["Stacking optimisé"] = stack_opt
res_opt = cv_table(best).sort_values("roc_auc", ascending=False)
res_opt[["accuracy", "precision", "recall", "f1", "roc_auc", "roc_auc_std", "F1 train", "écart train-val (F1)"]].round(4)
""")
code(r"""
cmp = pd.DataFrame({"F1 avant": res_base["f1"], "F1 après": res_opt["f1"],
                    "AUC avant": res_base["roc_auc"], "AUC après": res_opt["roc_auc"]}).dropna().sort_values("AUC après")
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, m in zip(axes, ["F1", "AUC"]):
    cmp[[f"{m} avant", f"{m} après"]].plot.barh(ax=ax, color=["#b9c7d8", "#5b8bd0"])
    ax.set_xlim(max(0, cmp[[f"{m} avant", f"{m} après"]].values.min() - 0.05), 1.0); ax.set_title(f"{m} en validation croisée")
plt.tight_layout(); savefig("avant_apres_optimisation"); plt.show()
""")
md(r"""
### 4.1 Choix du modèle final

On retient le modèle ayant la meilleure AUC en validation croisée ; en cas d'égalité
(écart < 0,002), on préfère le modèle le plus simple et celui dont l'écart
entraînement/validation est le plus faible (meilleur compromis biais/variance).
""")
code(r"""
complexity = {"Régression logistique": 1, "Naive Bayes": 1, "k-NN": 2, "Arbre de décision": 2, "SVM (RBF)": 3,
              "Random Forest": 4, "Gradient Boosting": 4, "Stacking optimisé": 5}
cand = res_opt.copy(); cand["complexité"] = cand.index.map(complexity)
near = cand[cand["roc_auc"] >= cand["roc_auc"].max() - 0.002].sort_values(["complexité", "écart train-val (F1)"])
FINAL_NAME = near.index[0]
final_model = best[FINAL_NAME]
print("Modèle final :", FINAL_NAME)
if FINAL_NAME in tuning.index:
    print("Hyperparamètres :", tuning.loc[FINAL_NAME, "meilleurs paramètres"])
RESULTS["finalModel"] = FINAL_NAME
RESULTS["finalCvAuc"] = res_opt.loc[FINAL_NAME, "roc_auc"]; RESULTS["finalCvFone"] = res_opt.loc[FINAL_NAME, "f1"]
RESULTS["finalCvAcc"] = res_opt.loc[FINAL_NAME, "accuracy"]
""")

# ---------------------------------------------------------------------------
md(r"""
## 5. Évaluation finale sur l'échantillon de test (cycles 2001 à 2205)

Le modèle final est réentraîné sur les 2000 cycles d'entraînement puis évalué **une seule
fois** sur les 205 cycles de test. Pour comparaison, tous les modèles optimisés sont aussi
évalués (ce tableau ne sert pas à choisir le modèle, le choix ayant été fait en 4.1).
""")
code(r"""
test_rows, test_proba = [], {}
for name, model in best.items():
    m = clone(model).fit(X_train, y_train)
    p = m.predict_proba(X_test)[:, 1]; pred = (p >= 0.5).astype(int)
    test_proba[name] = p
    if name == FINAL_NAME:
        final_fitted = m
    test_rows.append({"modèle": name, "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred, zero_division=0),
                      "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred), "roc_auc": roc_auc_score(y_test, p)})
res_test = pd.DataFrame(test_rows).set_index("modèle").sort_values("roc_auc", ascending=False)
res_test.round(4)
""")
code(r"""
pred_final = final_fitted.predict(X_test)
print(f"Modèle final : {FINAL_NAME}\n")
print(classification_report(y_test, pred_final, target_names=["non optimale", "optimale"], digits=4))
for k, v in res_test.loc[FINAL_NAME].items():
    RESULTS[f"test{k.replace('_', '').replace('f1', 'fone').capitalize()}"] = v

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ConfusionMatrixDisplay.from_predictions(y_test, pred_final, ax=axes[0], display_labels=["non optimale", "optimale"], cmap="Blues", colorbar=False)
axes[0].set_title(f"Matrice de confusion — test ({FINAL_NAME})")
for name in res_test.index:
    lw, a = (2.5, 1) if name == FINAL_NAME else (1, .6)
    fpr, tpr, _ = roc_curve(y_test, test_proba[name])
    axes[1].plot(fpr, tpr, lw=lw, alpha=a, label=f"{name} ({res_test.loc[name, 'roc_auc']:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", lw=.8); axes[1].legend(fontsize=8, loc="lower right")
axes[1].set_xlabel("taux de faux positifs"); axes[1].set_ylabel("taux de vrais positifs"); axes[1].set_title("Courbes ROC — test")
plt.tight_layout(); savefig("test_final"); plt.show()
cmat = confusion_matrix(y_test, pred_final); RESULTS.update(testTN=cmat[0, 0], testFP=cmat[0, 1], testFN=cmat[1, 0], testTP=cmat[1, 1])
""")
md("### 5.1 Analyse des erreurs")
code(r"""
err = profile.iloc[N_TRAIN:].copy()
err["proba_optimale"] = final_fitted.predict_proba(X_test)[:, 1]
err["prédiction"] = pred_final; err["réel"] = y_test.values; err["erreur"] = err["prédiction"] != err["réel"]
by_valve = err.groupby("valve").agg(cycles=("erreur", "size"), erreurs=("erreur", "sum"), taux_erreur=("erreur", "mean"))
by_stable = err.groupby("stable").agg(cycles=("erreur", "size"), erreurs=("erreur", "sum"), taux_erreur=("erreur", "mean"))
display(by_valve.round(3)); display(by_stable.round(3))
RESULTS["nTestErrors"] = int(err["erreur"].sum())

fig, ax = plt.subplots(figsize=(14, 3.5))
ax.scatter(err.index, err["proba_optimale"], c=np.where(err["réel"] == 1, "#3a9d5d", "#d0605e"), s=14)
ax.scatter(err.index[err["erreur"]], err.loc[err["erreur"], "proba_optimale"], facecolors="none", edgecolors="k", s=60, label="erreur")
ax.axhline(0.5, color="k", ls="--", lw=.8); ax.set_xlabel("cycle"); ax.set_ylabel("P(valve optimale)")
ax.set_title("Probabilités prédites sur le test (vert = réellement optimale, rouge = non optimale)"); ax.legend()
plt.tight_layout(); savefig("probas_test"); plt.show()
""")

md(r"""
## 6. Interprétation : quelles variables expliquent une valve non optimale ?

L'importance par permutation mesure la baisse d'AUC sur le jeu de test quand on mélange
aléatoirement une variable : elle est indépendante du type de modèle.
""")
code(r"""
pi = permutation_importance(final_fitted, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1)
imp = pd.DataFrame({"importance": pi.importances_mean, "std": pi.importances_std}, index=X_test.columns)
imp = imp[imp["importance"] > 0].sort_values("importance", ascending=False).head(15)
fig, ax = plt.subplots(figsize=(9, 5.5))
imp["importance"].iloc[::-1].plot.barh(ax=ax, xerr=imp["std"].iloc[::-1], color="#5b8bd0")
ax.set_xlabel("baisse moyenne d'AUC (permutation)"); ax.set_title(f"Importance par permutation — {FINAL_NAME}")
plt.tight_layout(); savefig("importance_permutation"); plt.show()
selected = final_fitted.named_steps["select"] if hasattr(final_fitted, "named_steps") else None
if selected is not None:
    names_after_corr = final_fitted.named_steps["corr"].get_feature_names_out()
    kept = names_after_corr[selected.get_support()]
    print(f"{len(kept)} variables utilisées par le modèle final :", list(kept))
    RESULTS["nFinalFeatures"] = len(kept)
imp.round(4)
""")

md(r"""
### 6.1 Analyse physique : l'instant de commutation de la valve

La variable dominante, `PS2_w1_mean`, est la pression moyenne entre 6 et 12 s, c'est-à-dire
autour de la remontée de pression visible vers 9,4 s sur la figure des profils moyens. On
mesure directement cet instant : premier instant après 5 s où PS2 dépasse 50 bar.
""")
code(r"""
seg_start = 500  # 5 s
rise_time = pd.Series((np.argmax(ps2[:, seg_start:] > 50, axis=1) + seg_start) / 100.0, index=profile.index, name="t_montee_s")
rt = pd.DataFrame({"t_montee_s": rise_time, "valve": profile["valve"], "accumulator": profile["accumulator"],
                   "jeu": np.where(profile.index <= N_TRAIN, "entraînement", "test")})
display(rt.groupby(["jeu", "valve"])["t_montee_s"].agg(["mean", "std", "min", "max"]).round(3))
display(rt[rt["jeu"] == "entraînement"].groupby(["accumulator", "valve"])["t_montee_s"].mean().unstack().round(3))

fig, ax = plt.subplots(figsize=(10, 4))
sns.stripplot(data=rt, x="valve", y="t_montee_s", hue="jeu", dodge=True, size=3, alpha=.6, ax=ax, palette=["#5b8bd0", "#d0605e"])
ax.set_xlabel("condition de la valve (%)"); ax.set_ylabel("instant de montée en pression (s)")
ax.set_title("Instant où PS2 dépasse 50 bar selon la condition de la valve")
plt.tight_layout(); savefig("instant_commutation"); plt.show()

means = rt[rt["jeu"] == "entraînement"].groupby("valve")["t_montee_s"].mean()
RESULTS["riseHundred"] = f"{means[100]:.2f}"; RESULTS["riseNinety"] = f"{means[90]:.2f}"
RESULTS["riseEighty"] = f"{means[80]:.2f}"; RESULTS["riseSeventythree"] = f"{means[73]:.2f}"
RESULTS["delayNinetyMs"] = f"{1000 * (means[90] - means[100]):.0f}"
""")
md(r"""
**Expérience a posteriori (ne remplace pas le modèle choisi en 4.1).** Un classifieur à
**une seule variable**, l'instant de montée en pression, est entraîné sur les 2000 premiers
cycles et évalué sur le test.
""")
code(r"""
one = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))])
cv_one = cross_validate(one, rt[["t_montee_s"]].iloc[:N_TRAIN], y_train, cv=CV, scoring=["f1", "roc_auc"])
one.fit(rt[["t_montee_s"]].iloc[:N_TRAIN], y_train)
p_one = one.predict(rt[["t_montee_s"]].iloc[N_TRAIN:])
print(f"CV : F1={cv_one['test_f1'].mean():.4f}  AUC={cv_one['test_roc_auc'].mean():.4f}")
print(f"Test : accuracy={accuracy_score(y_test, p_one):.4f}  F1={f1_score(y_test, p_one):.4f}")
RESULTS["oneFeatTestAcc"] = accuracy_score(y_test, p_one); RESULTS["oneFeatTestFone"] = f1_score(y_test, p_one)
""")

md("## 7. Sauvegarde du modèle et export des résultats pour le rapport")
code(r"""
MODEL_DIR = Path("../models") if Path("../models").exists() else Path("models"); MODEL_DIR.mkdir(exist_ok=True)
final_all = clone(best[FINAL_NAME]).fit(X_train, y_train)  # entraîné uniquement sur les 2000 premiers cycles
joblib.dump({"model": final_all, "name": FINAL_NAME, "features": list(X.columns), "n_train": N_TRAIN},
            MODEL_DIR / "modele_valve.joblib")
X.to_csv(MODEL_DIR / "features.csv.gz")  # utilisé par l'application web
profile.to_csv(MODEL_DIR / "profile.csv")
print("Modèle sauvegardé dans", MODEL_DIR / "modele_valve.joblib")

def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("{", r"\{").replace("}", r"\}")

def to_latex_table(df, path, fmt="{:.4f}"):
    cols = list(df.columns)
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             " & ".join(["Modèle"] + [tex_escape(c) for c in cols]) + r" \\", r"\midrule"]
    for idx, *row in df.itertuples():  # itertuples conserve le type de chaque colonne (entiers)
        vals = [fmt.format(v) if isinstance(v, (float, np.floating)) else tex_escape(v) for v in row]
        lines.append(" & ".join([tex_escape(idx)] + vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines), encoding="utf-8")

rename = {"accuracy": "Accuracy", "precision": "Précision", "recall": "Rappel", "f1": "F1", "roc_auc": "AUC"}
to_latex_table(res_base[["accuracy", "precision", "recall", "f1", "roc_auc", "F1 train", "écart train-val (F1)"]]
               .rename(columns={**rename, "écart train-val (F1)": "Écart F1"}), TAB_DIR / "cv_base.tex", "{:.3f}")
to_latex_table(res_opt[["accuracy", "precision", "recall", "f1", "roc_auc", "F1 train", "écart train-val (F1)"]]
               .rename(columns={**rename, "écart train-val (F1)": "Écart F1"}), TAB_DIR / "cv_opt.tex", "{:.3f}")
to_latex_table(res_test.rename(columns=rename), TAB_DIR / "test.tex", "{:.3f}")
to_latex_table(tuning[["méthode", "combinaisons", "AUC avant", "AUC après", "temps (s)"]]
               .rename(columns={"méthode": "Méthode", "combinaisons": "Comb.", "temps (s)": "Temps (s)"}), TAB_DIR / "tuning.tex", "{:.3f}")
to_latex_table(rob, TAB_DIR / "robustesse.tex", "{:.3f}")
to_latex_table(by_valve.rename(columns={"cycles": "Cycles", "erreurs": "Erreurs", "taux_erreur": "Taux"}).rename(index=lambda v: f"valve {v} %"),
               TAB_DIR / "erreurs_valve.tex", "{:.3f}")
with open(TAB_DIR / "params.tex", "w", encoding="utf-8") as fh:
    fh.write("\\begin{tabular}{lp{11cm}}\n\\toprule\nModèle & Meilleurs hyperparamètres \\\\\n\\midrule\n")
    for name, row in tuning.iterrows():
        p = ", ".join(f"{k}={v}" for k, v in row["meilleurs paramètres"].items())
        fh.write(f"{tex_escape(name)} & \\texttt{{{tex_escape(p)}}} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

def macro_name(k):
    return "res" + "".join(ch for ch in k if ch.isalpha())

with open(TAB_DIR / "resultats.tex", "w", encoding="utf-8") as fh:
    fh.write("% Fichier généré automatiquement par le notebook : ne pas modifier à la main\n")
    for k, v in RESULTS.items():
        if isinstance(v, (float, np.floating)):
            v = f"{v:.3f}" if abs(v) <= 1 else f"{v:.1f}"
        fh.write(f"\\newcommand{{\\{macro_name(k)}}}{{{tex_escape(v)}}}\n")
print(open(TAB_DIR / "resultats.tex", encoding="utf-8").read())
""")

md(r"""
## Conclusion

Le notebook couvre l'ensemble de la méthodologie demandée : contrôle qualité des données,
binarisation de la cible, analyse de l'équilibre des classes et des corrélations,
extraction et sélection de variables, comparaison de neuf modèles (linéaires, à base
d'instances, arbres, probabiliste, SVM, ensembles), évaluation par validation croisée
stratifiée avec matrices de confusion et courbes ROC, analyse biais/variance par courbes
d'apprentissage, optimisation par Grid Search et Random Search, puis évaluation unique sur
les 205 cycles de test. Le modèle final est sauvegardé dans `models/modele_valve.joblib` et
utilisé par l'application web (`app/`).
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = ROOT / "notebooks" / "projet_valve.ipynb"
nbf.write(nb, out)
print("Notebook écrit :", out)
