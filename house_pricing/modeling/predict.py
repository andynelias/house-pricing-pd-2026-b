"""Inferencia con el modelo ganador de House Pricing.

Expone el pipeline resultado (`models/best_model.joblib`) construido en los notebooks 2.0 y 3.0:
selección de features, ingeniería de características, modelo ganador y reversión de `log1p` a
dólares. Está pensado para que FastAPI importe `load_model`, `predict` y `predict_records`, y
también se puede ejecutar desde la línea de comandos sobre un CSV crudo:

    python -m house_pricing.modeling.predict --input-path data/raw/test.csv

Cada ejecución registra su log en esta misma carpeta, en `predict_<YYYYMMDD_HHMMSS>.log`.
"""

from datetime import datetime
from functools import lru_cache
from pathlib import Path
import time
from typing import Any

import joblib
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
import typer

from house_pricing.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

MODEL_PATH = MODELS_DIR / "best_model.joblib"
LOG_DIR = Path(__file__).resolve().parent
ID_COLUMN = "Id"
PREDICTION_COLUMN = "SalePrice"
VALID_COLUMN = "is_valid"

# Rango de precios coherentes: [0.5 x mínimo, 2 x máximo] del SalePrice de train.csv.
TRAIN_MIN_PRICE = 34_900.0
TRAIN_MAX_PRICE = 755_000.0
PRICE_LOWER_BOUND = 0.5 * TRAIN_MIN_PRICE
PRICE_UPPER_BOUND = 2.0 * TRAIN_MAX_PRICE

_log_file: Path | None = None

app = typer.Typer()


def configure_logging(log_dir: Path = LOG_DIR) -> Path:
    """Agrega un archivo de log con estampa de tiempo, una sola vez por proceso.

    Args:
        log_dir: Carpeta donde se crea el archivo de log.

    Returns:
        Ruta del archivo de log en uso.
    """
    global _log_file
    if _log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_file = log_dir / f"predict_{timestamp}.log"
        logger.add(_log_file, level="DEBUG", encoding="utf-8", enqueue=True)
        logger.info(f"Log de inferencia en {_log_file}")
    return _log_file


@lru_cache(maxsize=1)
def load_model(model_path: Path = MODEL_PATH) -> TransformedTargetRegressor:
    """Carga el pipeline resultado una sola vez y lo reutiliza en llamadas posteriores.

    Args:
        model_path: Ruta del archivo joblib con el pipeline resultado.

    Returns:
        Pipeline ajustado que recibe datos crudos y devuelve precios en dólares.

    Raises:
        FileNotFoundError: Si el archivo del modelo no existe.
    """
    configure_logging()
    if not model_path.exists():
        logger.error(f"No existe el modelo en {model_path}")
        raise FileNotFoundError(f"No existe el modelo en {model_path}")

    model = joblib.load(model_path)
    estimator = model.regressor_.named_steps["model"]
    logger.info(f"Modelo cargado: {type(estimator).__name__} desde {model_path}")
    return model


def get_required_features(model: TransformedTargetRegressor) -> tuple[list[str], list[str]]:
    """Obtiene del pipeline las columnas requeridas y cuáles de ellas son numéricas.

    Args:
        model: Pipeline resultado ajustado.

    Returns:
        Tupla con la lista de columnas requeridas y la lista de columnas numéricas.
    """
    pipeline = model.regressor_
    required = list(pipeline.named_steps["select"].transformers_[0][2])
    fe_steps = pipeline.named_steps["fe"].named_steps
    numeric = list(fe_steps["median_imputer"].variables_)
    numeric += list(fe_steps["random_sample_imputer"].variables_)
    return required, numeric


def get_seed_features(model: TransformedTargetRegressor) -> list[str]:
    """Obtiene las columnas que el imputador aleatorio usa como semilla de cada observación.

    Args:
        model: Pipeline resultado ajustado.

    Returns:
        Lista de columnas que no pueden venir vacías.
    """
    fe_steps = model.regressor_.named_steps["fe"].named_steps
    return list(fe_steps["random_sample_imputer"].random_state)


def validate_input(data: pd.DataFrame, model: TransformedTargetRegressor) -> pd.DataFrame:
    """Valida la entrada y normaliza los tipos de datos que espera el pipeline.

    Las columnas numéricas se convierten a float y las categóricas a texto, conservando los
    valores faltantes (el pipeline los imputa). Las columnas que no usa el modelo se ignoran.

    Args:
        data: Datos crudos con una fila por casa.
        model: Pipeline resultado ajustado.

    Returns:
        Copia de los datos con los tipos normalizados.

    Raises:
        ValueError: Si la entrada está vacía, faltan columnas requeridas, hay valores no
            numéricos en columnas numéricas o faltan valores en las columnas semilla.
    """
    if data.empty:
        logger.error("La entrada está vacía")
        raise ValueError("La entrada no tiene filas")

    required, numeric = get_required_features(model)
    missing_columns = [column for column in required if column not in data.columns]
    if missing_columns:
        logger.error(f"Faltan columnas requeridas: {missing_columns}")
        raise ValueError(f"Faltan columnas requeridas: {missing_columns}")

    ignored = [column for column in data.columns if column not in required + [ID_COLUMN]]
    if ignored:
        logger.debug(f"{len(ignored)} columnas no usadas por el modelo se ignoran")

    if ID_COLUMN in data.columns and data[ID_COLUMN].duplicated().any():
        logger.warning(f"Hay {data[ID_COLUMN].duplicated().sum()} {ID_COLUMN} duplicados")

    clean = data.copy()
    for column in required:
        if column in numeric:
            converted = pd.to_numeric(clean[column], errors="coerce")
            invalid = clean[column].notna() & converted.isna()
            if invalid.any():
                examples = clean.loc[invalid, column].head(3).tolist()
                logger.error(f"Valores no numéricos en {column}: {examples}")
                raise ValueError(f"La columna {column} debe ser numérica; valores: {examples}")
            clean[column] = converted.astype(float)
        else:
            clean[column] = clean[column].map(
                lambda value: value if pd.isna(value) else str(value)
            )

    seed_features = get_seed_features(model)
    seed_missing = clean[seed_features].isna().sum()
    if seed_missing.any():
        detail = seed_missing[seed_missing > 0].to_dict()
        logger.error(f"Valores faltantes en columnas semilla: {detail}")
        raise ValueError(f"Las columnas {seed_features} no pueden tener faltantes: {detail}")

    missing_values = int(clean[required].isna().sum().sum())
    if missing_values:
        logger.info(f"{missing_values} valores faltantes en la entrada; el pipeline los imputa")
    return clean


def flag_coherent_predictions(predictions: np.ndarray) -> np.ndarray:
    """Marca las predicciones coherentes: finitas, positivas y dentro del rango esperado.

    Args:
        predictions: Precios predichos en dólares.

    Returns:
        Arreglo booleano, `True` donde la predicción es coherente.
    """
    predictions = np.asarray(predictions, dtype=float)
    finite = np.isfinite(predictions)
    in_range = (predictions >= PRICE_LOWER_BOUND) & (predictions <= PRICE_UPPER_BOUND)
    return finite & (predictions > 0) & in_range


def predict(data: pd.DataFrame, model: TransformedTargetRegressor | None = None) -> pd.DataFrame:
    """Genera predicciones de SalePrice a partir de datos crudos.

    Args:
        data: Datos crudos con las columnas que usa el modelo (por ejemplo, filas de test.csv).
        model: Pipeline resultado; si es `None` se carga `models/best_model.joblib`.

    Returns:
        DataFrame con `Id` (o el índice de la fila si la entrada no lo trae), `SalePrice` en
        dólares e `is_valid`, que indica si la predicción es coherente.

    Raises:
        ValueError: Si la entrada no es válida (ver `validate_input`).
    """
    configure_logging()
    model = model if model is not None else load_model()
    start = time.perf_counter()
    logger.info(f"Inicio de inferencia: {len(data)} filas de entrada")

    clean = validate_input(data, model)
    predictions = model.predict(clean)
    is_valid = flag_coherent_predictions(predictions)

    ids = clean[ID_COLUMN].to_numpy() if ID_COLUMN in clean.columns else clean.index.to_numpy()
    result = pd.DataFrame({ID_COLUMN: ids, PREDICTION_COLUMN: predictions, VALID_COLUMN: is_valid})

    n_rows, n_valid = len(result), int(is_valid.sum())
    for _, row in result.loc[~result[VALID_COLUMN]].iterrows():
        logger.warning(
            f"Predicción no coherente para {ID_COLUMN}={row[ID_COLUMN]}: "
            f"{row[PREDICTION_COLUMN]} fuera de [{PRICE_LOWER_BOUND:,.0f}, "
            f"{PRICE_UPPER_BOUND:,.0f}] o no finita"
        )
    if n_rows != len(data):
        logger.error(f"Filas de salida ({n_rows}) distintas a las de entrada ({len(data)})")
        raise RuntimeError("La cantidad de filas de salida no coincide con la de entrada")

    logger.info(
        f"Fin de inferencia: {n_valid}/{n_rows} filas coherentes ({n_valid / n_rows:.1%}), "
        f"{n_rows - n_valid} no coherentes, {time.perf_counter() - start:.3f}s"
    )
    return result


def predict_records(
    records: list[dict[str, Any]], model: TransformedTargetRegressor | None = None
) -> list[dict[str, Any]]:
    """Genera predicciones desde registros tipo JSON (formato que usará FastAPI).

    Args:
        records: Lista de diccionarios, uno por casa, con las columnas que usa el modelo.
        model: Pipeline resultado; si es `None` se carga `models/best_model.joblib`.

    Returns:
        Lista de diccionarios con `Id`, `SalePrice` e `is_valid`. Las predicciones no finitas se
        devuelven como `None` para que sean serializables a JSON.

    Raises:
        ValueError: Si la entrada no es válida (ver `validate_input`).
    """
    result = predict(pd.DataFrame.from_records(records), model=model)
    result[PREDICTION_COLUMN] = result[PREDICTION_COLUMN].astype(object)
    result.loc[~np.isfinite(result[PREDICTION_COLUMN].astype(float)), PREDICTION_COLUMN] = None
    return [
        {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in row.items()
        }
        for row in result.to_dict(orient="records")
    ]


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "test.csv",
    output_path: Path = PROCESSED_DATA_DIR / "predictions_script.csv",
    model_path: Path = MODEL_PATH,
) -> None:
    """Ejecuta la inferencia sobre un CSV crudo y guarda Id, SalePrice e is_valid en un CSV.

    Args:
        input_path: CSV crudo de entrada (por defecto `data/raw/test.csv`).
        output_path: CSV de salida con las predicciones.
        model_path: Ruta del pipeline resultado.
    """
    configure_logging()
    logger.info(f"Leyendo datos de {input_path}")
    data = pd.read_csv(input_path)
    result = predict(data, model=load_model(model_path))
    result.to_csv(output_path, index=False)
    logger.success(f"Predicciones guardadas en {output_path}")


if __name__ == "__main__":
    app()
