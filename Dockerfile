# Image unique pour : exécuter le notebook, lancer les tests et servir l'application web.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 VALVE_DATA_DIR=/app/data
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY valve_ml/ valve_ml/
COPY app/ app/
COPY tests/ tests/
COPY notebooks/ notebooks/
COPY tools/ tools/
COPY models/ models/
COPY pytest.ini .
RUN mkdir -p data rapport/figures rapport/tables

EXPOSE 5000
# Par défaut : application web. Voir le README pour les autres commandes.
CMD ["python", "app/app.py"]
