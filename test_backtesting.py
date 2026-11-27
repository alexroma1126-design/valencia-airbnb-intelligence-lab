from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting import (
    SETTINGS,
    calculate_metrics,
    get_origin,
    select_model
)

ROOT = Path.cwd()
OUT = ROOT / "results" / "tables"

passed = 0


def check(name, condition):

    global passed

    if not condition:
        raise AssertionError(name)

    passed += 1
    print(f"[OK] {name}")


print("=" * 70)
print("AUDITORIA DE BACKTESTING")
print("=" * 70)

val = pd.read_csv(
    OUT / "forecast_validation_metrics.csv"
)

test = pd.read_csv(
    OUT / "forecast_test_metrics.csv"
)

pred = pd.read_csv(
    OUT / "forecast_test_predictions.csv",
    parse_dates=["fecha", "origen"]
)

check(
    "Existen resultados de validacion",
    not val.empty
)

check(
    "Existen resultados de test",
    not test.empty
)

check(
    "Existen predicciones",
    not pred.empty
)

check(
    "Origen anterior al objetivo",
    (pred["origen"] < pred["fecha"]).all()
)

check(
    "Predicciones no negativas",
    pred["prediccion"].dropna().ge(0).all()
)

check(
    "Valores reales no negativos",
    pred["real"].ge(0).all()
)

check(
    "Sin predicciones temporales duplicadas",
    not pred.duplicated([
        "barrio",
        "frecuencia",
        "fecha"
    ]).any()
)

check(
    "MAE no negativo",
    test["MAE"].dropna().ge(0).all()
)

check(
    "RMSE mayor o igual que MAE",
    (
        test["RMSE"] + 1e-9
        >= test["MAE"]
    ).all()
)

for frequency, cfg in SETTINGS.items():

    check(
        f"Validacion y test separados: {frequency}",
        pd.Timestamp(cfg["validation_end"])
        < pd.Timestamp(cfg["test_start"])
    )

    selected = pred[
        pred["frecuencia"] == frequency
    ]

    expected = selected["fecha"].map(
        lambda d: get_origin(
            d,
            frequency,
            cfg["h"]
        )
    )

    check(
        f"Horizonte correcto: {frequency}",
        expected.eq(selected["origen"]).all()
    )

    check(
        f"Fechas dentro del test: {frequency}",
        selected["fecha"].between(
            pd.Timestamp(cfg["test_start"]),
            pd.Timestamp(cfg["test_end"])
        ).all()
    )

correct = True

for _, row in test.iterrows():

    candidates = val[
        (val["barrio"] == row["barrio"])
        & (
            val["frecuencia"]
            == row["frecuencia"]
        )
    ]

    winner = select_model(candidates)

    if winner is None:

        correct = False
        break

    if (
        winner["modelo"]
        != row["modelo_seleccionado"]
    ):

        correct = False
        break

check(
    "Seleccion basada solo en validacion",
    correct
)

m = calculate_metrics(
    [10, 20, 30],
    [12, 18, 33]
)

check(
    "Formula MAE correcta",
    np.isclose(m["MAE"], 7 / 3)
)

check(
    "Formula RMSE correcta",
    np.isclose(m["RMSE"], np.sqrt(17 / 3))
)

check(
    "Formula WAPE correcta",
    np.isclose(m["WAPE"], 100 * 7 / 60)
)

print("\n" + "=" * 70)
print("PRUEBAS SUPERADAS:", passed)
print("BACKTESTING VALIDADO")
print("=" * 70)

print("\nRESULTADOS PRINCIPALES")

columns = [
    "barrio",
    "frecuencia",
    "modelo_seleccionado",
    "MAE",
    "RMSE",
    "WAPE",
    "mejora_MAE_pct",
    "cobertura_test"
]

print(
    test.loc[
        test["barrio"].isin([
            "RUSSAFA",
            "CABANYAL-CANYAMELAR",
            "BENIFERRI",
            "TODOS"
        ]),
        columns
    ].round(3).to_string(index=False)
)
