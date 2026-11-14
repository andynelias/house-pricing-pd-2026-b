"""Pruebas de `house_pricing.modeling.predict` con observaciones aleatorias de test.csv."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor

from house_pricing.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from house_pricing.modeling.predict import (
    PRICE_LOWER_BOUND,
    PRICE_UPPER_BOUND,
    flag_coherent_predictions,
    get_required_features,
    get_seed_features,
    load_model,
    predict,
    predict_records,
)

SAMPLE_SIZE = 25
SEEDS = [7, 42, 2026]


@pytest.fixture(scope="session")
def model() -> TransformedTargetRegressor:
    """Pipeline resultado cargado una sola vez para todas las pruebas."""
    return load_model()


@pytest.fixture(scope="session")
def test_raw() -> pd.DataFrame:
    """Datos crudos de test.csv."""
    return pd.read_csv(RAW_DATA_DIR / "test.csv")


@pytest.mark.parametrize("seed", SEEDS)
def test_random_sample_predictions_are_coherent(
    model: TransformedTargetRegressor, test_raw: pd.DataFrame, seed: int
) -> None:
    """Una muestra aleatoria de test.csv produce una predicción coherente por fila."""
    sample = test_raw.sample(n=SAMPLE_SIZE, random_state=seed)

    result = predict(sample, model=model)

    assert len(result) == len(sample)
    assert result["Id"].tolist() == sample["Id"].tolist()
    assert result["is_valid"].all()
    assert result["SalePrice"].between(PRICE_LOWER_BOUND, PRICE_UPPER_BOUND).all()


@pytest.mark.parametrize("seed", SEEDS)
def test_random_sample_matches_inference_notebook(
    model: TransformedTargetRegressor, test_raw: pd.DataFrame, seed: int
) -> None:
    """El script reproduce las predicciones del notebook 4.0 (predictions.csv)."""
    sample = test_raw.sample(n=SAMPLE_SIZE, random_state=seed)
    notebook = pd.read_csv(PROCESSED_DATA_DIR / "predictions.csv").set_index("Id")

    result = predict(sample, model=model)

    expected = notebook.loc[result["Id"], "SalePrice"].to_numpy()
    np.testing.assert_allclose(result["SalePrice"].to_numpy(), expected, rtol=1e-9)


def test_predict_records_returns_json_ready_rows(
    model: TransformedTargetRegressor, test_raw: pd.DataFrame
) -> None:
    """`predict_records` acepta registros tipo JSON y devuelve tipos nativos de Python."""
    sample = test_raw.sample(n=5, random_state=0)
    records = sample.astype(object).where(sample.notna(), None).to_dict(orient="records")

    result = predict_records(records, model=model)

    assert len(result) == len(records)
    assert [row["Id"] for row in result] == sample["Id"].tolist()
    for row in result:
        assert set(row) == {"Id", "SalePrice", "is_valid"}
        assert isinstance(row["Id"], int)
        assert isinstance(row["SalePrice"], float)
        assert row["is_valid"] is True


def test_single_record_with_missing_values(
    model: TransformedTargetRegressor, test_raw: pd.DataFrame
) -> None:
    """Un registro con faltantes en sus columnas numéricas (salvo las semilla) se predice."""
    _, numeric = get_required_features(model)
    seeds = get_seed_features(model)
    record = test_raw.sample(n=1, random_state=1).astype(object).iloc[0].to_dict()
    record.update({column: None for column in numeric if column not in seeds})

    result = predict_records([record], model=model)

    assert len(result) == 1
    assert result[0]["SalePrice"] is not None


def test_prediction_does_not_depend_on_batch(
    model: TransformedTargetRegressor, test_raw: pd.DataFrame
) -> None:
    """Una casa con faltantes imputados al azar recibe el mismo precio sola o en un lote."""
    with_missing = test_raw[test_raw["LotFrontage"].isna()]
    house = with_missing.iloc[[5]]
    batch = pd.concat([with_missing.iloc[:5], house])

    alone = predict(house, model=model)["SalePrice"].iloc[0]
    in_batch = predict(batch, model=model)["SalePrice"].iloc[-1]

    assert alone == pytest.approx(in_batch, rel=1e-12)


def test_missing_seed_value_raises(
    model: TransformedTargetRegressor, test_raw: pd.DataFrame
) -> None:
    """Un faltante en una columna semilla del imputador aleatorio lanza ValueError."""
    sample = test_raw.sample(n=3, random_state=0)
    sample.iloc[0, sample.columns.get_loc(get_seed_features(model)[0])] = np.nan

    with pytest.raises(ValueError, match="no pueden tener faltantes"):
        predict(sample, model=model)


def test_empty_input_raises(model: TransformedTargetRegressor, test_raw: pd.DataFrame) -> None:
    """Una entrada sin filas lanza ValueError."""
    with pytest.raises(ValueError, match="no tiene filas"):
        predict(test_raw.iloc[0:0], model=model)


def test_missing_required_columns_raises(
    model: TransformedTargetRegressor, test_raw: pd.DataFrame
) -> None:
    """Si faltan columnas requeridas se lanza ValueError indicando cuáles."""
    sample = test_raw.sample(n=3, random_state=0).drop(columns=["GrLivArea", "Neighborhood"])

    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        predict(sample, model=model)


def test_non_numeric_value_raises(
    model: TransformedTargetRegressor, test_raw: pd.DataFrame
) -> None:
    """Un texto en una columna numérica lanza ValueError."""
    sample = test_raw.sample(n=3, random_state=0)
    sample["GrLivArea"] = sample["GrLivArea"].astype(object)
    sample.iloc[0, sample.columns.get_loc("GrLivArea")] = "grande"

    with pytest.raises(ValueError, match="debe ser numérica"):
        predict(sample, model=model)


def test_flag_coherent_predictions() -> None:
    """Solo las predicciones finitas, positivas y dentro del rango son coherentes."""
    predictions = np.array(
        [180_000.0, np.nan, np.inf, -5.0, PRICE_LOWER_BOUND - 1, PRICE_UPPER_BOUND + 1]
    )

    flags = flag_coherent_predictions(predictions)

    assert flags.tolist() == [True, False, False, False, False, False]
