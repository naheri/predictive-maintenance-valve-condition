"""Application web : prédiction de la condition de la valve pour un numéro de cycle.

Lancement : python app/app.py  (puis http://localhost:5000)
API JSON  : GET /api/predict?cycle=2050
Le modèle et les caractéristiques sont produits par le notebook (dossier models/).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from valve_ml.selection import CorrelationFilter  # noqa: E402

# Le notebook définit CorrelationFilter dans __main__ : on l'y expose pour pouvoir dépickler le modèle.
import __main__  # noqa: E402

__main__.CorrelationFilter = CorrelationFilter
MODEL_DIR = Path(os.environ.get("VALVE_MODEL_DIR", ROOT / "models"))

app = Flask(__name__)
_STATE: dict = {}


def load_artifacts():
    if not _STATE:
        bundle = joblib.load(MODEL_DIR / "modele_valve.joblib")
        _STATE["bundle"] = bundle
        _STATE["features"] = pd.read_csv(MODEL_DIR / "features.csv.gz", index_col="cycle")[bundle["features"]]
        _STATE["profile"] = pd.read_csv(MODEL_DIR / "profile.csv", index_col="cycle")
    return _STATE


def predict_cycle(cycle: int) -> dict:
    st = load_artifacts()
    X, profile, bundle = st["features"], st["profile"], st["bundle"]
    if cycle not in X.index:
        raise KeyError(f"Le cycle doit être compris entre {X.index.min()} et {X.index.max()}.")
    proba = float(bundle["model"].predict_proba(X.loc[[cycle]])[0, 1])
    valve = int(profile.loc[cycle, "valve"])
    return {
        "cycle": cycle,
        "jeu": "entraînement" if cycle <= bundle["n_train"] else "test",
        "proba_optimale": round(proba, 4),
        "prediction": "optimale" if proba >= 0.5 else "non optimale",
        "valve_reelle_pct": valve,
        "verite": "optimale" if valve == 100 else "non optimale",
        "correct": (proba >= 0.5) == (valve == 100),
        "modele": bundle["name"],
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result, error, cycle = None, None, request.values.get("cycle", "")
    if cycle:
        try:
            result = predict_cycle(int(cycle))
        except (ValueError, KeyError) as exc:
            error = str(exc).strip("'\"") if isinstance(exc, KeyError) else "Numéro de cycle invalide."
    n = len(load_artifacts()["features"])
    return render_template("index.html", result=result, error=error, cycle=cycle, n_cycles=n)


@app.route("/api/predict")
def api_predict():
    try:
        return jsonify(predict_cycle(int(request.args["cycle"])))
    except KeyError as exc:
        return jsonify({"erreur": str(exc).strip("'\"")}), 400
    except ValueError:
        return jsonify({"erreur": "Numéro de cycle invalide."}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
