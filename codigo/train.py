"""Entrena dos modelos y registra parámetros, métricas y modelos en MLflow."""
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

datos = pd.read_csv(Path("datos") / "datos_limp.csv")
X, y = datos.drop(columns="target"), datos["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
modelos = {
    "Regresion_logistica": (Pipeline([("escala", StandardScaler()), ("modelo", LogisticRegression(max_iter=2000, random_state=42))]), {"modelo__C": [0.1, 1, 10]}),
    "Random_Forest": (Pipeline([("modelo", RandomForestClassifier(random_state=42, n_jobs=-1))]), {"modelo__n_estimators": [100, 200], "modelo__max_depth": [None, 6]})
}
mlflow.set_tracking_uri("file:" + str(Path("mlruns").resolve()))
mlflow.set_experiment("Actividad5_Comparacion")
for nombre, (modelo, parametros) in modelos.items():
    ajuste = GridSearchCV(modelo, parametros, scoring="f1", cv=cv, n_jobs=-1)
    ajuste.fit(X_train, y_train)
    pred = ajuste.predict(X_test)
    prob = ajuste.predict_proba(X_test)[:, 1]
    with mlflow.start_run(run_name=nombre):
        mlflow.log_params({k: str(v) for k, v in ajuste.best_params_.items()})
        mlflow.log_metrics({"accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred), "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred), "roc_auc": roc_auc_score(y_test, prob)})
        mlflow.sklearn.log_model(ajuste.best_estimator_, name="modelo")
        print(nombre, "F1:", round(f1_score(y_test, pred), 4))
