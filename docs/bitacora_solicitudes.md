# Bitácora de solicitudes — Proyecto House Pricing

Este archivo lleva el registro cronológico de lo que se le ha pedido a Claude en este proyecto,
para tener trazabilidad del avance sin depender del historial de chat. Cada vez que se agregue
una solicitud nueva, se debe añadir una entrada al final siguiendo el mismo formato.

**Proyecto:** Predicción de precios de casas (Kaggle — *House Prices: Advanced Regression Techniques*)
**Autor:** Anderson Elias

---

## 1. Estructura de carpetas (Cookiecutter Data Science)
- Se solicitó la estructura de carpetas recomendada por Cookiecutter Data Science.
- Se generó el proyecto dentro de `house_pricing/` usando la herramienta oficial `ccds`
  (versión moderna del template, PyPI: `cookiecutter-data-science`).
- Configuración elegida:
  - Entorno/dependencias: **conda** + `requirements.txt`
  - Linting/formato: **ruff**
  - Testing: **pytest**
  - Scaffold de código: **sí** (`dataset.py`, `features.py`, `modeling/train.py`, `modeling/predict.py`, `plots.py`)
  - Licencia: ninguna (proyecto privado/académico)
  - Python: 3.12
  - Autor: Anderson Elias
- El archivo `hola.py` que ya existía en la carpeta se conservó sin cambios.

## 2. Descarga del dataset de Kaggle
- Se solicitó descargar el dataset de la competencia de Kaggle
  [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
  y colocar train y test en `data/raw/`.
- Al no tener credenciales de la API de Kaggle configuradas, el usuario descargó el `.zip`
  manualmente desde el navegador y se extrajo/limpió en el proyecto.
- Archivos colocados en `data/raw/`:
  - `train.csv` (1,460 filas, incluye `SalePrice`)
  - `test.csv` (1,459 filas, sin `SalePrice`)
  - `data_description.txt`
  - `sample_submission.csv`

## 3. Primer notebook de análisis exploratorio (EDA)
- Se solicitó crear el primer notebook de exploración de datos en `notebooks/`, nombrado
  `analisis_exploratorio`, siguiendo la convención de nombres de Cookiecutter Data Science:
  `<paso>.<subpaso>-<iniciales>-<descripcion-corta>.ipynb`.
- Archivo creado: **`notebooks/1.0-ae-analisis-exploratorio.ipynb`**
- Se indicó que, de ahora en adelante, todos los notebooks nuevos deben seguir esta misma
  convención de nombres.
- Contenido inicial del notebook:
  1. Carga de `train.csv` y `test.csv` desde `data/raw/` (vía `house_pricing.config.RAW_DATA_DIR`)
  2. Info general y tipos de datos (`info()`, `describe()`)
  3. Análisis de valores faltantes (tabla + gráfico de barras)
  4. Distribución de `SalePrice` (normal y log-transformada)
  5. Correlaciones numéricas con `SalePrice` + heatmap de las 10 variables más correlacionadas
  6. Sección de "próximos pasos"

## 4. Corrección de error al ejecutar el notebook
- El notebook fallaba al ejecutarse por falta de las librerías `seaborn` y `loguru`.
- Se instalaron ambas librerías en el entorno y se volvió a ejecutar el notebook completo
  sin errores (`train shape: (1460, 81)`, `test shape: (1459, 80)`).

## 5. Scatter plots de variables numéricas vs `SalePrice` + tabla de correlaciones
- Se solicitó agregar, en una celda nueva (con encabezado y descripción), un scatter plot por
  cada variable numérica contra `SalePrice`, generado **en un ciclo**, mostrando el valor de la
  correlación en el título de cada gráfico, y un DataFrame con las correlaciones ordenadas de
  mayor a menor.
- Se agregó la sección correspondiente al notebook con una grilla de scatter plots y el
  DataFrame `corr_df`.

## 6. Separación de variables discretas vs continuas
- Se notó que algunas gráficas se veían discontinuas (en bandas verticales), por tratarse de
  variables discretas/ordinales.
- Se solicitó separar las variables numéricas en dos grupos (**discretas** y **continuas**),
  repetir los scatter plots solo para las continuas, y para las discretas hacer un **boxplot**
  (eje x = variable, eje y = `SalePrice`) mostrando además los puntos individuales (usando
  `seaborn`).
- Se implementó un umbral de clasificación: variables con **25 o menos valores únicos** se
  consideran discretas (18 variables), el resto continuas (18 variables).
- Resultado en el notebook:
  - Sección 6: clasificación discretas/continuas + `corr_df` con columna `tipo`
  - Sección 7: scatter plots solo de las 18 variables continuas vs `SalePrice`
  - Sección 8: boxplots de las 18 variables discretas vs `SalePrice`, con `sns.boxplot`
    (`fliersize=0`) + `sns.stripplot` (jitter) superpuesto para mostrar los puntos individuales

## 7. Bitácora de solicitudes (este archivo)
- Se solicitó guardar el avance del proyecto en algún archivo para tener registro de lo pedido
  hasta el momento. Se creó este archivo: `docs/bitacora_solicitudes.md`.

## 8. Eliminación del scatter combinado de variables continuas
- Se solicitó eliminar la sección de scatter plots que mostraba todas las variables continuas
  juntas en una sola grilla (sección 7 del notebook), ya que no era necesaria.
- Se eliminó la celda markdown "Dispersión de variables continuas vs `SalePrice`" y su celda de
  código correspondiente del notebook.
- Se renumeraron las secciones siguientes: "Boxplots de variables discretas" pasó de sección 8 a
  **sección 7**, y "Próximos pasos" pasó a **sección 8**.
- El notebook se volvió a ejecutar completo sin errores tras el cambio.

## 9. Recomendaciones sobre sesgo de variables continuas (guardadas, no aplicadas)
- Se pidió una recomendación (sin implementar) para tratar el sesgo de las variables continuas
  y mejorar la forma de su relación con `SalePrice`.
- Por decisión del usuario se guardaron para la etapa de creación de features en
  `docs/recomendaciones_feature_engineering.md` (outliers, `log1p` del target, indicadoras para
  variables zero-inflated, Box-Cox/Yeo-Johnson, validación con Spearman/LOWESS).

## 10. Boxplots de variables categóricas con correlation ratio (η)
- Se solicitó un boxplot por cada variable categórica vs `SalePrice` con una métrica similar a
  la correlación en el título de cada gráfica.
- Se agregó la **sección 8** del notebook: 43 variables categóricas, boxplot + `stripplot`,
  categorías ordenadas por mediana de `SalePrice` y `NaN` tratados como categoría `"NA"`
  (significan ausencia según `data_description.txt`).
- Métrica: **correlation ratio η** (raíz de la proporción de varianza de `SalePrice` explicada
  por las medias de cada categoría, rango 0–1). Se implementó con pandas/numpy, sin librerías
  extra. Se incluye el DataFrame `eta_df` ordenado de mayor a menor con número de categorías.
- Top η: `Neighborhood` 0.74, `ExterQual` 0.69, `BsmtQual` 0.68, `KitchenQual` 0.68.
- "Próximos pasos" pasó a ser la **sección 9**.

## 11. Selección de features para el modelo (top 10 por grupo)
- Se solicitó seleccionar las 10 mejores variables continuas, discretas y categóricas según las
  métricas calculadas, como features para construir el modelo, en forma de arreglo.
- Criterio: |Pearson| con `SalePrice` para continuas y discretas; η para categóricas.
- Se agregó la **sección 9** del notebook ("Próximos pasos" pasó a la **sección 10**):
  - `top_continuous = ['GrLivArea', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MasVnrArea', 'BsmtFinSF1', 'LotFrontage']`
  - `top_discrete = ['OverallQual', 'GarageCars', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'HalfBath', 'BsmtFullBath', 'BedroomAbvGr', 'KitchenAbvGr', 'PoolArea']`
  - `top_categorical = ['Neighborhood', 'ExterQual', 'BsmtQual', 'KitchenQual', 'GarageFinish', 'FireplaceQu', 'Foundation', 'GarageType', 'BsmtFinType1', 'HeatingQC']`
  - `selected_features` = unión de los tres (30 features).

## 12. Configuración del pipeline y traspaso de sesión
- Se solicitó crear `pipeline_config.yaml` con la variable `MODEL_FEATURES` conteniendo las 30
  features seleccionadas, para usarlo en el siguiente paso.
- Se creó `pipeline_config.yaml` en la raíz del proyecto (validado: carga con `yaml.safe_load`,
  30 features, todas existen en `train.csv`, sin duplicados).
- Se creó `docs/contexto_siguiente_sesion.md` con el estado del proyecto, convenciones,
  conclusiones del EDA, pendientes y siguiente paso sugerido.
- Se guardó en la memoria de Claude una referencia a ese documento.

## 13. Ingeniería de características (notebook 2.0)
- Se solicitó la FE en 5 pasos (imputación de categóricas, codificación, outliers,
  transformación de continuas, escalado), en celdas separadas con títulos y descripciones,
  usando un **pipeline de sklearn** con transformadores de **feature-engine**, y generar el
  dataset de entrada para el notebook de modelos. Indicación: consultar en vez de asumir.
- Se instaló `feature-engine` 1.9.4. Decisiones tomadas por el usuario:
  - Faltantes: >20% en train → descartar (numéricas y categóricas; se descartó `FireplaceQu`,
    47%). Numéricas: ≤5% → mediana, 5–20% → `RandomSampleImputer`. Categóricas → moda.
  - Codificación: ordinal manual para calidad (`ExterQual`, `BsmtQual`, `KitchenQual`,
    `HeatingQC`; Po=0 … Ex=4) y one-hot para nominales.
  - Outliers: eliminar las 2 casas con `GrLivArea > 4000` y precio bajo (Id 524 y 1299) +
    `Winsorizer` IQR 1.5 cola derecha en las 7 continuas de área.
  - Transformación: `YeoJohnsonTransformer` en `GrLivArea`, `TotalBsmtSF`, `1stFlrSF`,
    `MasVnrArea`, `BsmtFinSF1`, `LotFrontage`.
  - Escalado: `StandardScaler` en numéricas y ordinales; dummies quedan en 0/1.
  - Target: `SalePrice_log = log1p(SalePrice)`. Split train/val 80/20 (`random_state=42`).
    Salida en CSV.
- Archivos generados:
  - `notebooks/2.0-ae-ingenieria-de-caracteristicas.ipynb` (secciones 0–12)
  - `data/processed/train_fe.csv` (1166 × 72), `val_fe.csv` (292 × 72), `test_fe.csv` (1459 × 71):
    `Id` + 70 features (24 numéricas escaladas + 46 dummies) + `SalePrice_log`
  - `models/fe_pipeline.joblib` (pipeline ajustado; verificado que reproduce `test_fe.csv`)
- Correcciones durante la ejecución: la banda 5–20% se había aplicado también a
  `GarageFinish`/`GarageType` (se corrigió a moda), y Yeo-Johnson se fijó a la lista de 6
  acordada (con el umbral calculado en el split, `TotalBsmtSF` habría quedado fuera, skew 0.45).

## 14. Entrenamiento de modelos — torneo (notebook 3.0)
- Se solicitó un notebook de entrenamiento con KFold k=10 y optimización de hiperparámetros
  (≥3 configuraciones), un torneo de ≥5 modelos (regresiones, ensambles y redes neuronales),
  comparación en validación con RMSE real y RMSE normalizado al rango, gráficas de desempeño por
  modelo, y evitar sobreajuste y fuga de datos. Indicación: no hacer nada más y consultar dudas.
- Decisiones del usuario: pipeline de FE **dentro** del CV (se parte de `data/raw` con el mismo
  split, verificado contra `train_fe.csv`/`val_fe.csv`, y se clona `fe_pipeline.joblib` sin
  ajustar); `GridSearchCV`; RMSE en escala log para elegir configuración; modelos Ridge, Lasso,
  ElasticNet, RandomForest, HistGradientBoosting y MLPRegressor (39 configuraciones); NRMSE con
  el rango de `SalePrice` en validación; gráfica predicho vs real.
- Archivo: `notebooks/3.0-ae-entrenamiento-de-modelos.ipynb` (secciones 0–7). No se guardaron
  modelos ni submission.
- Resultado del torneo (validación):

  | # | Modelo | RMSE real | NRMSE | RMSE val (log) | RMSE train (log) | RMSE CV (log) |
  |---|---|---|---|---|---|---|
  | 1 | ElasticNet | 22,346 | 0.0536 | 0.1328 | 0.1215 | 0.1287 |
  | 2 | Lasso | 22,357 | 0.0537 | 0.1328 | 0.1215 | 0.1287 |
  | 3 | Ridge | 22,434 | 0.0538 | 0.1330 | 0.1214 | 0.1287 |
  | 4 | HistGradientBoosting | 23,369 | 0.0561 | 0.1432 | 0.0913 | 0.1343 |
  | 5 | RandomForest | 24,089 | 0.0578 | 0.1492 | 0.0517 | 0.1371 |
  | 6 | MLP | 108,216 | 0.2597 | 0.3048 | 0.1501 | 0.2767 |
- Observaciones pendientes de decisión: varios mejores hiperparámetros quedaron en el borde de
  su grilla; RandomForest muestra brecha train/CV grande (sobreajuste); el MLP generaliza mal
  (4 predicciones fuera de rango, una de 1.68 M).

## 15. Guardado del modelo ganador y notebook de inferencia
- Se solicitó guardar el modelo ganador en `models/` junto con el pipeline completo de FE, y un
  notebook de inferencia que use los datos crudos de test y genere las predicciones con un
  pipeline resultado.
- Decisiones del usuario: reentrenar con train + val (1458 casas); revertir `log1p` dentro del
  pipeline (`TransformedTargetRegressor`); salida en formato submission de Kaggle; nombres
  `models/best_model.joblib` y `notebooks/4.0-ae-inferencia.ipynb`.
- Notebook 3.0, nueva **sección 8**: construye `TransformedTargetRegressor(Pipeline(select → fe →
  model))`. `select` es un `ColumnTransformer` que conserva las 30 `MODEL_FEATURES` para que el
  pipeline acepte los CSV crudos completos. El ganador se toma dinámicamente del torneo
  (ElasticNet, alpha=0.0005, l1_ratio=0.2).
- Notebook `4.0-ae-inferencia.ipynb` (secciones 0–4): carga `best_model.joblib` y `test.csv` crudo,
  predice en dólares, valida contra `sample_submission.csv` (columnas, `Id`, sin NaN, precios > 0)
  y guarda `data/processed/predictions.csv` (1459 filas, `Id`, `SalePrice`).
- Predicciones de test: media 178,014, mediana 157,704, rango 55,486 – 518,025.

## 16. Script de inferencia para FastAPI
- Se solicitó un script de inferencia que use el pipeline resultado, con firmas tipadas y
  docstrings, PEP 8, logging de calidad entrada/salida con archivo con estampa de tiempo, y un
  mecanismo de prueba con observaciones aleatorias de test.
- Decisiones del usuario: `house_pricing/modeling/predict.py` (logs en esa misma carpeta);
  funciones para DataFrame y para registros JSON; coherente = finita, > 0 y dentro de
  [17,450 ; 1,510,000] (0.5 × mín y 2 × máx de train); entrada inválida → excepción, filas no
  coherentes → WARNING e `is_valid=False`; loguru; pytest; CLI con typer; líneas de 99.
- Se instalaron `pytest`, `typer` y `ruff`. Se eliminó `tests/test_data.py` (placeholder con
  `assert False`) y se agregó `[tool.pytest.ini_options]` a `pyproject.toml`.
- `predict.py`: `configure_logging`, `load_model` (con caché), `get_required_features`,
  `get_seed_features`, `validate_input`, `flag_coherent_predictions`, `predict`,
  `predict_records` y `main` (CLI: `python -m house_pricing.modeling.predict`, por defecto
  `test.csv` → `data/processed/predictions_script.csv`). Validado con ruff (reglas E, W, F, I, N,
  D con convención Google).
- `tests/test_predict.py`: 14 pruebas (muestras aleatorias de 25 filas con 3 semillas, igualdad con
  el notebook 4.0, registros JSON, faltantes, independencia del lote, errores de entrada y regla de
  coherencia). Resultado: 14 passed.
- **Hallazgo y corrección:** el `RandomSampleImputer` con semilla global hacía que el precio de una
  casa con `LotFrontage`/`GarageYrBlt` faltante dependiera del lote (Id 1513: 148,630 sola vs
  143,284 en lote). Por decisión del usuario se cambió a `seed="observation"` con semilla
  `YearBuilt + OverallQual + GrLivArea` (columnas que pasan a ser obligatorias sin faltantes). Se
  regeneraron los notebooks 2.0, 3.0 y 4.0: ElasticNet sigue ganando con los mismos
  hiperparámetros, RMSE real 22,363 (antes 22,346).

## 17. Cierre de sesión: contexto para la siguiente
- Se solicitó guardar todo lo necesario para la siguiente sesión.
- Se reescribió `docs/contexto_siguiente_sesion.md` con el estado vigente: tabla del torneo
  actualizada tras el cambio de semilla, archivos clave, convenciones, comandos para retomar, orden
  de regeneración de notebooks, uso de `predict.py` desde FastAPI, pendientes y notas técnicas del
  entorno Windows.
- Nota: la tabla de la entrada 14 corresponde a la primera ejecución del torneo; la vigente está en
  la sección 7 del documento de contexto.
- Se actualizó la memoria de Claude (estado del proyecto y notas técnicas).

---

> **Nota:** Añadir una nueva entrada numerada a este archivo cada vez que se complete una
> solicitud relevante del proyecto.
