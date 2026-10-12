# Maintenance prédictive : condition de la valve d'un banc hydraulique

Projet de Machine Learning (EFREI, ING2 BDML, 2026/2027). Pour chaque cycle de 60 s, le
modèle prédit si la valve est **optimale (100 %)** ou non, à partir de la pression **PS2**
(100 Hz), du débit **FS1** (10 Hz) et du fichier **profile.txt**. Les 2000 premiers cycles
servent à l'apprentissage, les 205 derniers au test final.

## Arborescence

```
projet_ml/
├── data/                    # PS2.txt, FS1.txt, profile.txt (à déposer, non versionnés)
├── notebooks/projet_valve.ipynb   # notebook complet (livrable)
├── valve_ml/                # module : chargement, caractéristiques, sélection, modèles
├── tests/                   # tests unitaires pytest
├── app/                     # application web Flask (prédiction pour un cycle)
├── models/                  # modèle final + caractéristiques (générés par le notebook)
├── rapport/                 # rapport LaTeX (figures/ et tables/ générées par le notebook,
│                            #   captures/ = vos captures d'écran)
├── tools/build_notebook.py  # régénère le notebook à partir du module
├── Dockerfile
└── requirements.txt
```

## Utilisation

```bash
pip install -r requirements.txt
# 1. Déposer PS2.txt, FS1.txt et profile.txt dans data/ (ou VALVE_DATA_DIR=/chemin)
# 2. Exécuter le notebook (génère figures, tableaux, modèle)
cd notebooks && jupyter nbconvert --to notebook --execute projet_valve.ipynb --output projet_valve.ipynb && cd ..
# 3. Compiler le rapport
cd rapport && pdflatex rapport.tex && pdflatex rapport.tex && cd ..
# 4. Tests et application web
python -m pytest -q
python app/app.py            # http://localhost:5000  —  API : /api/predict?cycle=2050
```

Sous **Google Colab** : téléverser le notebook, créer `/content/data` avec les trois fichiers,
puis exécuter toutes les cellules.

## Docker

```bash
docker build -t valve-ml .
docker run --rm -p 5000:5000 valve-ml                  # application web
docker run --rm valve-ml python -m pytest -q            # tests
```

## Captures d'écran du rapport

Chaque cadre rouge « CAPTURE À INSÉRER » du PDF indique le nom de fichier attendu
(ex. `rapport/captures/appli_web.png`). Déposez l'image avec ce nom puis recompilez : elle
remplace automatiquement le cadre.
