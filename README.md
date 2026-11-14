# house_pricing

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Predicción del precio de venta de casas (`SalePrice`) con machine learning, a partir de la
competencia de Kaggle
[House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

**Autor:** Anderson Elias

## Resultados

Torneo de 6 modelos con `GridSearchCV` y KFold k=10. La ingeniería de características va dentro
del CV para evitar fuga de datos. La comparación final se hace sobre un set de validación (20% de
train):

| # | Modelo | RMSE real (USD) | RMSE normalizado al rango |
|---|---|---|---|
| 1 | **ElasticNet** | **22,363** | **0.0537** |
| 2 | Lasso | 22,371 | 0.0537 |
| 3 | Ridge | 22,469 | 0.0539 |
| 4 | HistGradientBoosting | 23,423 | 0.0562 |
| 5 | RandomForest | 23,920 | 0.0574 |
| 6 | MLP | 101,113 | 0.2427 |

El modelo ganador se guarda como un pipeline completo (selección de features → ingeniería de
características → ElasticNet → reversión de `log1p`) que recibe datos crudos y devuelve el precio
en dólares.

## Cómo reproducir

### 1. Dependencias

Python 3.12. `requirements.txt` todavía no lista todas las dependencias; mientras se actualiza:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn feature-engine pyyaml joblib loguru \
    python-dotenv tqdm typer jupyter pytest ruff
```

### 2. Datos (no incluidos en el repositorio)

Los datos de Kaggle no se redistribuyen. Descarga el `.zip` desde la
[página de datos de la competencia](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
(hay que aceptar sus reglas) y coloca en `data/raw/`:

- `train.csv`
- `test.csv`
- `data_description.txt`
- `sample_submission.csv`

### 3. Notebooks (en orden)

Los modelos entrenados (`models/*.joblib`) tampoco se versionan: se regeneran ejecutando los
notebooks en este orden.

| Notebook | Qué hace | Genera |
|---|---|---|
| `1.0-ae-analisis-exploratorio` | EDA y selección de las 30 features | — (las features están en `pipeline_config.yaml`) |
| `2.0-ae-ingenieria-de-caracteristicas` | Pipeline de FE (sklearn + feature-engine) | `data/processed/*_fe.csv`, `models/fe_pipeline.joblib` |
| `3.0-ae-entrenamiento-de-modelos` | Torneo de modelos y guardado del ganador | `models/best_model.joblib` |
| `4.0-ae-inferencia` | Predicciones para el test de Kaggle | `data/processed/predictions.csv` |

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/2.0-ae-ingenieria-de-caracteristicas.ipynb
```

### 4. Inferencia y pruebas

```bash
python -m house_pricing.modeling.predict   # test.csv -> data/processed/predictions_script.csv
python -m pytest                           # requiere data/raw y haber ejecutado los notebooks
```

Desde código (por ejemplo, una API con FastAPI):

```python
from house_pricing.modeling.predict import load_model, predict_records

model = load_model()
result = predict_records(records, model)  # [{"Id": ..., "SalePrice": ..., "is_valid": ...}]
```

Cada ejecución deja un log en `house_pricing/modeling/predict_<YYYYMMDD_HHMMSS>.log` con el
control de calidad entrada/salida.

## Estructura del proyecto

```
├── Makefile                <- Comandos de conveniencia de cookiecutter
├── README.md
├── pipeline_config.yaml    <- MODEL_FEATURES: las 30 features seleccionadas en el EDA
├── pyproject.toml          <- Metadatos del paquete y configuración de ruff y pytest
├── requirements.txt
├── data                    <- No versionada (ver "Datos")
│   ├── raw                 <- Datos originales de Kaggle
│   └── processed           <- Datasets transformados y predicciones
├── docs
│   ├── bitacora_solicitudes.md               <- Registro cronológico del desarrollo
│   ├── contexto_siguiente_sesion.md          <- Estado actual, decisiones y pendientes
│   └── recomendaciones_feature_engineering.md
├── models                  <- Pipelines entrenados (*.joblib, no versionados)
├── notebooks               <- Convención: <paso>.<subpaso>-<iniciales>-<descripcion>
├── references
├── reports
│   └── figures
├── tests
│   └── test_predict.py     <- Pruebas con observaciones aleatorias de test.csv
└── house_pricing           <- Código fuente
    ├── config.py           <- Rutas del proyecto
    ├── dataset.py          <- (scaffold, sin implementar)
    ├── features.py         <- (scaffold, sin implementar)
    ├── plots.py            <- (scaffold, sin implementar)
    └── modeling
        ├── predict.py      <- Inferencia con el modelo ganador (funciones + CLI)
        └── train.py        <- (scaffold, sin implementar)
```

Las decisiones de diseño y el detalle de cada etapa están en `docs/contexto_siguiente_sesion.md`
y `docs/bitacora_solicitudes.md`.
