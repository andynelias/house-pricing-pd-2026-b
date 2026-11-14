# Recomendaciones pendientes para Feature Engineering

Recomendaciones surgidas durante el EDA (`notebooks/1.0-ae-analisis-exploratorio.ipynb`), a
aplicar cuando se construya el mecanismo de creación de features (`house_pricing/features.py`).
**Estado:** puntos 1–4 aplicados en `notebooks/2.0-ae-ingenieria-de-caracteristicas.ipynb` (excepto las indicadoras binarias de variables zero-inflated, que no se pidieron). Puntos 5–6 pendientes.

---

## 1. Sesgo en variables continuas

Skewness medido en `train.csv`:

| Variable | Skewness |
|---|---|
| `LotArea` | 12.21 |
| `BsmtFinSF2` | 4.26 |
| `ScreenPorch` | 4.12 |
| `EnclosedPorch` | 3.09 |
| `MasVnrArea` | 2.67 |
| `OpenPorchSF` | 2.36 |
| `LotFrontage` | 2.16 |
| `SalePrice` (target) | 1.88 |
| `BsmtFinSF1` | 1.69 |
| `WoodDeckSF` | 1.54 |
| `TotalBsmtSF` | 1.52 |
| `1stFlrSF` | 1.38 |
| `GrLivArea` | 1.37 |
| `BsmtUnfSF` | 0.92 |
| `2ndFlrSF` | 0.81 |
| `GarageArea` | 0.18 |
| `YearRemodAdd`, `YearBuilt`, `GarageYrBlt` | entre -0.5 y -0.65 |

### Secuencia recomendada
1. **Outliers primero:** revisar `LotArea` y `GrLivArea` (casos conocidos con `GrLivArea > 4000`
   y precio bajo). Winsorizar o eliminar antes de decidir la transformación.
2. **Target:** modelar sobre `np.log1p(SalePrice)`; además de normalizarlo, linealiza relaciones
   log-log con predictores sesgados. Revertir con `np.expm1` al predecir.
3. **Variables zero-inflated** (`BsmtFinSF2`, `ScreenPorch`, `EnclosedPorch`, `OpenPorchSF`,
   `WoodDeckSF`, `MasVnrArea`): crear indicadoras binarias (`HasScreenPorch`, `HasMasVnr`, ...) y
   transformar la magnitud con Yeo-Johnson (tolera ceros) o solo para valores > 0.
4. **Continuas genuinas sesgadas** (`LotArea`, `GrLivArea`, `TotalBsmtSF`, `1stFlrSF`,
   `LotFrontage`, `BsmtFinSF1`): `np.log1p` como primera opción; `PowerTransformer`
   (Box-Cox / Yeo-Johnson) si se quiere un exponente óptimo.
5. **Validar la forma de la relación:** agregar LOWESS a los scatter (`sns.regplot(..., lowess=True)`)
   y comparar Pearson vs Spearman; Spearman mucho mayor que Pearson indica relación monótona
   no lineal (candidata a log).
6. **Re-evaluar después de transformar:** skew antes/después, histogramas y scatter vs target.
