# Actividad 5 — Preparación de datos, entrenamiento y comparación de modelos de inteligencia artificial

Dataset: Breast Cancer Wisconsin (Diagnostic), accesible mediante `sklearn.datasets.load_breast_cancer`. 569 registros, 30 variables numéricas y una etiqueta binaria (0=maligno, 1=benigno). Uso académico exclusivamente.

## Reproducir
1. Abre `fuentes/entrena.ipynb` en Google Colab y ejecuta las celdas en orden.
2. El notebook instala dependencias, exporta `datos/datos_ini.csv` y `datos/datos_limp.csv`, entrena regresión logística y Random Forest, ajusta GridSearchCV con cinco folds y registra dos modelos en MLflow.
3. Consulta la tabla `mlflow.search_runs()` y guarda capturas para `evidencias_mlflow/`.
4. Alternativamente, desde la raíz: `pip install -r requirements.txt`, `python fuentes/datos_prep.py`, `python fuentes/train.py`.

## Decisiones
No se eliminaron registros arbitrariamente: el dataset original no tiene nulos ni duplicados exactos. La estandarización de variables se aplica solo a regresión logística y dentro del pipeline de CV para evitar fuga de información. La clase positiva (1) corresponde a benigno; por tanto, recall positivo no mide directamente la sensibilidad de casos malignos.

## Versionamiento
El original se conserva sin modificaciones en `datos_ini.csv` y el resultado de deduplicación en `datos_limp.csv`. Consultar `CHANGELOG.md`; Git/GitHub registra las versiones mediante commits.

## Conclusión
Comparar métricas de prueba y validación cruzada antes de elegir un modelo. No usar como sistema clínico. Los resultados locales se encuentran en `resultados/`; los registros MLflow deben verificarse ejecutando el notebook.

## Evidencias finales
El notebook ejecutado se encuentra en fuentes/entrena.ipynb; las capturas MLflow en evidencias_mlflow/ y los resultados de Colab en resultados/. Se incluyen 4 runs (2 por algoritmo) por repetición de ejecución. El informe final se encuentra en Reporte_Tecnico_FINAL.docx. El repositorio GitHub debe publicarse desde la cuenta del estudiante.
