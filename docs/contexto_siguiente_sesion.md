# Contexto para la siguiente sesión — House Pricing

Documento de traspaso: resume el estado actual del proyecto y lo necesario para retomar el trabajo.
Última actualización: 2026-11-09 (cierre de sesión, tras crear el script de inferencia).

---

## 1. El proyecto
- **Objetivo:** predecir `SalePrice` (precio de venta de casas) — competencia de Kaggle
  *House Prices: Advanced Regression Techniques*. Problema de **regresión**.
- **Autor:** Anderson Elias (iniciales `ae` en los notebooks).
- **Estructura:** Cookiecutter Data Science v2 (`ccds`), módulo Python `house_pricing/`.
- **Entorno:** Python 3.12 en Windows (sin entorno conda activo; todo instalado en el Python del
  usuario). Librerías en uso: pandas, numpy, matplotlib, seaborn, scikit-learn 1.9.1,
  feature-engine 1.9.4, pyyaml, joblib, loguru, typer, pytest, ruff, jupyter. También está
  instalado tensorflow 2.21 (no se usa). **No instalados:** fastapi, xgboost, lightgbm, optuna.

## 2. Estado actual
| Etapa | Estado |
|---|---|
| EDA (`1.0-ae-analisis-exploratorio.ipynb`) | Terminado |
| Selección de features (`pipeline_config.yaml`) | Terminado (30 features) |
| Ingeniería de características (`2.0-ae-ingenieria-de-caracteristicas.ipynb`) | Terminado |
| Torneo de modelos (`3.0-ae-entrenamiento-de-modelos.ipynb`) | Terminado; gana ElasticNet (RMSE real 22,363) |
| Modelo ganador (`models/best_model.joblib`) | Terminado (ElasticNet reentrenado con train + val) |
| Inferencia en notebook (`4.0-ae-inferencia.ipynb`) | Terminado → `data/processed/predictions.csv` |
| Script de inferencia (`house_pricing/modeling/predict.py`) + pruebas | Terminado (14 pruebas pytest OK, ruff OK) |
| **API con FastAPI** | **Siguiente paso probable** (el usuario anticipó que consumirá `predict.py`) |
| Ajustes del torneo (sección 8) | Pendiente de decisión del usuario |
| Envío de `predictions.csv` a Kaggle | No realizado |

## 3. Archivos clave
| Archivo | Para qué sirve |
|---|---|
| `data/raw/train.csv` / `test.csv` | Datos originales (1460 × 81 / 1459 × 80) |
| `data/raw/data_description.txt` | Significado de cada columna y categoría |
| `data/raw/sample_submission.csv` | Formato de envío de Kaggle |
| `pipeline_config.yaml` | `MODEL_FEATURES`: las 30 features seleccionadas en el EDA |
| `notebooks/1.0-ae-analisis-exploratorio.ipynb` | EDA (secciones 1–10) |
| `notebooks/2.0-ae-ingenieria-de-caracteristicas.ipynb` | Pipeline de FE (secciones 0–12); genera los CSV `*_fe` y `fe_pipeline.joblib` |
| `notebooks/3.0-ae-entrenamiento-de-modelos.ipynb` | Torneo (secciones 0–7) y guardado del ganador (sección 8) |
| `notebooks/4.0-ae-inferencia.ipynb` | `test.csv` crudo → `best_model.joblib` → `predictions.csv` |
| `models/fe_pipeline.joblib` | Pipeline de FE ajustado con el 80% de train (el notebook 3.0 lo clona sin ajustar) |
| `models/best_model.joblib` | **Pipeline resultado:** `TransformedTargetRegressor(Pipeline(select → fe → ElasticNet))`; recibe CSV crudo y devuelve dólares |
| `data/processed/train_fe.csv`, `val_fe.csv`, `test_fe.csv` | Datasets transformados (1166 / 292 / 1459 filas; `Id` + 70 features + `SalePrice_log`) |
| `data/processed/predictions.csv` | Submission de Kaggle (`Id`, `SalePrice`), del notebook 4.0 |
| `data/processed/predictions_script.csv` | Salida de la CLI de `predict.py` (`Id`, `SalePrice`, `is_valid`); idéntica en precios a `predictions.csv` |
| `house_pricing/modeling/predict.py` | Inferencia para FastAPI + CLI typer; logs `predict_<timestamp>.log` en esa misma carpeta |
| `tests/test_predict.py` | 14 pruebas pytest con muestras aleatorias de test |
| `house_pricing/config.py` | Rutas (`RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, `MODELS_DIR`, `PROJ_ROOT`, ...) |
| `house_pricing/dataset.py`, `features.py`, `plots.py`, `modeling/train.py` | Scaffold de ccds, **sin implementar** |
| `docs/bitacora_solicitudes.md` | Registro cronológico de las 17 solicitudes |
| `docs/recomendaciones_feature_engineering.md` | Recomendaciones de sesgo del EDA (aplicadas en parte) |

## 4. Convenciones acordadas
- **Notebooks:** `<paso>.<subpaso>-ae-<descripcion-con-guiones>.ipynb`. El siguiente sería `5.0-ae-...`.
- Cada celda de notebook lleva **título markdown + descripción simple** de lo que hace; cada paso en
  su propia celda.
- Gráficas por variable se generan **en un ciclo** en grilla de subplots, con la métrica en el título.
- **Consultar al usuario ante cualquier decisión no definida; no asumir.** Agrupar dudas en
  preguntas con opción recomendada y aplicar exactamente lo acordado.
- Si el usuario dice "no hagas nada más", limitarse estrictamente a lo pedido.
- Código Python: PEP 8 con líneas de 99 (config de ruff del proyecto), firmas tipadas y docstrings
  estilo Google.
- Actualizar `docs/bitacora_solicitudes.md` con una entrada por cada solicitud completada.
- Respuestas y documentación en **español**.

## 5. Cómo retomar (comandos, desde la raíz `house_pricing/`)
```bash
python -m pytest                                    # 14 pruebas de predict.py
python -m house_pricing.modeling.predict            # inferencia CLI: test.csv -> predictions_script.csv
python -m ruff check --select E,W,F,I,N,D --config "lint.pydocstyle.convention='google'" house_pricing/modeling/predict.py tests/test_predict.py
jupyter nbconvert --to notebook --execute --inplace notebooks/<notebook>.ipynb
```
- **Orden de regeneración:** si cambia la FE, reejecutar en orden 2.0 → 3.0 (~4–8 min, reentrena
  y guarda `best_model.joblib`) → 4.0, y luego `python -m pytest` (una prueba compara contra
  `predictions.csv`).
- Uso desde FastAPI:
```python
from house_pricing.modeling.predict import load_model, predict_records

model = load_model()                      # se carga una vez (lru_cache)
result = predict_records(records, model)  # records: list[dict] con columnas crudas
# -> [{"Id": ..., "SalePrice": float | None, "is_valid": bool}, ...]
```

## 6. Pipeline de FE
Antes del pipeline se eliminan del train las 2 casas anómalas (Id 524 y 1299: `GrLivArea > 4000` con
precio bajo). Split train/val 80/20 con `random_state=42`. Target `SalePrice_log = log1p(SalePrice)`.
Pasos (todos de feature-engine salvo los envueltos en `SklearnTransformerWrapper`):
1. `DropFeatures`: descarta `FireplaceQu` (47% de NaN; regla: >20% → descartar).
2. `CategoricalImputer` (moda) en las 9 categóricas restantes.
3. `RandomSampleImputer` en `GarageYrBlt` y `LotFrontage` (5–20% de NaN) con **semilla por
   observación** (`YearBuilt + OverallQual + GrLivArea`): cada casa recibe siempre el mismo valor
   imputado, sin importar el lote. Esas 3 columnas no pueden venir vacías.
4. `MeanMedianImputer` (mediana) en el resto de numéricas (≤5%; cubre NaN que solo aparecen en test).
5. `OrdinalEncoder` (sklearn) en `ExterQual`, `BsmtQual`, `KitchenQual`, `HeatingQC` (Po=0 … Ex=4).
6. `OneHotEncoder` en `Neighborhood`, `GarageFinish`, `Foundation`, `GarageType`, `BsmtFinType1`.
7. `Winsorizer` IQR 1.5, cola derecha, en `GrLivArea`, `GarageArea`, `TotalBsmtSF`, `1stFlrSF`,
   `MasVnrArea`, `BsmtFinSF1`, `LotFrontage`.
8. `YeoJohnsonTransformer` en `GrLivArea`, `TotalBsmtSF`, `1stFlrSF`, `MasVnrArea`, `BsmtFinSF1`,
   `LotFrontage`.
9. `StandardScaler` (sklearn) en las 24 numéricas; las 46 dummies quedan en 0/1.

## 7. Torneo de modelos (resultado vigente)
KFold k=10 (shuffle, `random_state=42`), `GridSearchCV` (39 configuraciones), elección por RMSE en
escala log, FE **dentro** del CV (sin fuga). Comparación en validación:

| # | Modelo | RMSE real | NRMSE | RMSE val (log) | RMSE train (log) | RMSE CV (log) | Mejores hiperparámetros |
|---|---|---|---|---|---|---|---|
| 1 | ElasticNet | 22,363 | 0.0537 | 0.1332 | 0.1217 | 0.1287 | alpha=0.0005, l1_ratio=0.2 |
| 2 | Lasso | 22,371 | 0.0537 | 0.1332 | 0.1217 | 0.1287 | alpha=0.0001 |
| 3 | Ridge | 22,469 | 0.0539 | 0.1335 | 0.1216 | 0.1287 | alpha=1 |
| 4 | HistGradientBoosting | 23,423 | 0.0562 | 0.1403 | 0.0915 | 0.1348 | lr=0.03, max_depth=3, max_iter=600 |
| 5 | RandomForest | 23,920 | 0.0574 | 0.1476 | 0.0523 | 0.1378 | max_depth=15, max_features=0.33, min_samples_leaf=1 |
| 6 | MLP | 101,113 | 0.2427 | 0.3007 | 0.1504 | 0.2829 | alpha=0.1, hidden=(64,) |

NRMSE = RMSE real / rango de `SalePrice` en validación (416,639).

**Observaciones pendientes de decisión del usuario:**
- Hiperparámetros en el borde de su grilla: Lasso alpha=0.0001, ElasticNet alpha=0.0005 y
  l1_ratio=0.2, HistGB lr=0.03 y max_iter=600, RandomForest min_samples_leaf=1.
- RandomForest sobreajusta (train 0.052 vs CV 0.138 en log); HistGB en menor medida.
- MLP generaliza mal, con predicciones extremas fuera de rango; causa probable: target sin escalar.
- `best_model.joblib` se reentrenó con train + val: no tiene set reservado; su referencia es el RMSE
  del torneo.

## 8. Script de inferencia (`house_pricing/modeling/predict.py`)
- Funciones: `configure_logging`, `load_model` (cacheada), `get_required_features`,
  `get_seed_features`, `validate_input`, `flag_coherent_predictions`, `predict(df) -> DataFrame`,
  `predict_records(list[dict]) -> list[dict]`, `main` (CLI typer con `--input-path`,
  `--output-path`, `--model-path`).
- Columnas requeridas y tipos se leen del pipeline guardado (no depende de `pipeline_config.yaml`).
  Columnas extra se ignoran; `Id` es opcional (si falta, se usa el índice).
- Coherente = finita, > 0 y dentro de [17,450 ; 1,510,000] (0.5 × mín y 2 × máx de train).
- Errores (`ValueError`): entrada vacía, columnas requeridas faltantes, texto en columnas
  numéricas, faltantes en columnas semilla. `RuntimeError` si salen distintas filas de las que
  entran. Filas no coherentes → WARNING en el log e `is_valid=False`, sin detener el resto.
- Log por proceso en `house_pricing/modeling/predict_<YYYYMMDD_HHMMSS>.log` (se acumulan; no hay
  limpieza automática).

## 9. Pendientes y advertencias
1. **`requirements.txt` desactualizado:** solo lista `loguru`; faltan `seaborn`, `pyyaml`,
   `feature-engine`, `scikit-learn`, `joblib`, `typer`, `pytest`, `ruff` (y `fastapi` cuando se use).
2. **Colinealidad** entre features (`GarageCars`/`GarageArea`, `TotalBsmtSF`/`1stFlrSF`,
   `GrLivArea`/`TotRmsAbvGrd`, `YearBuilt`/`GarageYrBlt`) y one-hot completo (sin categoría
   eliminada): no afecta a los modelos regularizados ni a los árboles.
3. **Features débiles:** `PoolArea`, `KitchenAbvGr`, `BedroomAbvGr`.
4. **Zero-inflated** (`MasVnrArea`, `BsmtFinSF1`): sus indicadoras binarias (recomendación del EDA)
   no se crearon.
5. El proyecto **no es un repositorio git**.
6. `hola.py` en la raíz es un archivo previo del usuario; no se ha tocado.

## 10. Notas técnicas del entorno (Windows)
- Usar `python`, no `python3` (este último es el alias de Microsoft Store y falla).
- Para imprimir caracteres como η o tildes en consola: `PYTHONIOENCODING=utf-8`.
- Los heredocs de bash convierten `\\n` / `\f` en caracteres reales: los scripts que generan o
  editan notebooks deben escribirse como archivo (herramienta Write), no por heredoc.
- Los notebooks del repo son la **fuente de verdad**; los scripts que los generaron vivían en una
  carpeta temporal y no se conservan. Para cambios, editar el `.ipynb` (JSON) y reejecutar.
- Los `KeyError: ...joblib_memmapping_folder...` al ejecutar `GridSearchCV(n_jobs=-1)` son avisos de
  limpieza de joblib en Windows; no afectan los resultados.
