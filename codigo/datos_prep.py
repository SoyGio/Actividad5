"""Carga y revisión sencilla del dataset público de scikit-learn."""
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer

def preparar(carpeta="datos"):
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    original = load_breast_cancer(as_frame=True).frame
    original.to_csv(carpeta / "datos_ini.csv", index=False)
    limpio = original.drop_duplicates().copy()
    limpio.to_csv(carpeta / "datos_limp.csv", index=False)
    return limpio

if __name__ == "__main__":
    datos = preparar()
    print(f"Registros preparados: {len(datos)}")
