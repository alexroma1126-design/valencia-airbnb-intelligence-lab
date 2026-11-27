from pathlib import Path
import numpy as np
import pandas as pd

from src.uncertainty import (
    empirical_radius,
    prediction_interval,
    temporal_windows
)

from src.backtesting import get_origin

OUT = Path.cwd() / "results" / "tables"

passed = 0


def check(name, condition):

    global passed

    if not condition:

        raise AssertionError(name)

    passed += 1

    print(f"[OK] {name}")


print("=" * 72)
print("AUDITORIA DE INTERVALOS DE PREDICCION")
print("=" * 72)

check(
    "Radio correcto con errores constantes",
    np.isclose(
        empirical_radius([5] * 30),
        5
    )
)

check(
    "Muestra insuficiente controlada",
    np.isnan(
        empirical_radius([1, 2, 3])
    )
)

check(
    "Radio con 19 observaciones",
    np.isclose(
        empirical_radius(range(1, 20)),
        19
    )
)

check(
    "Limite inferior no negativo",
    prediction_interval(2, 5) == (0, 7)
)

check(
    "Intervalo simetrico correcto",
    prediction_interval(24, 5) == (19, 29)
)

check(
    "Redondeo conservador",
    prediction_interval(24.5, 3.2) == (21, 28)
)

for frequency, horizon in [
    ("semanal", 9),
    ("mensual", 3)
]:

    windows = temporal_windows(
        frequency,
        horizon
    )

    first_cal_origin = get_origin(
        windows["calibration_start"],
        frequency,
        horizon
    )

    check(
        f"Seleccion anterior a calibracion: {frequency}",
        windows["validation_end"]
        <= first_cal_origin
    )

    check(
        f"Calibracion anterior al test: {frequency}",
        windows["calibration_end"]
        <= windows["first_test_origin"]
    )

    check(
        f"Sin solapamiento de objetivos: {frequency}",
        windows["validation_end"]
        < windows["calibration_start"]
        < windows["calibration_end"]
        < windows["test_start"]
    )

final = pd.read_csv(
    OUT / "forecast_final_predictions.csv"
)

test = pd.read_csv(
    OUT / "forecast_interval_test.csv"
)

cal = pd.read_csv(
    OUT / "forecast_calibration.csv"
)

valid = final.dropna(
    subset=[
        "inferior_95",
        "superior_95"
    ]
)

check(
    "Existen pronosticos finales",
    not final.empty
)

check(
    "Existen intervalos disponibles",
    not valid.empty
)

check(
    "Predicciones no negativas",
    final["prediccion"].ge(0).all()
)

check(
    "Limites inferiores no negativos",
    valid["inferior_95"].ge(0).all()
)

check(
    "Intervalos contienen el pronostico",
    (
        (valid["inferior_95"] <= valid["prediccion"])
        & (
            valid["prediccion"]
            <= valid["superior_95"]
        )
    ).all()
)

check(
    "Nivel nominal del 95 por ciento",
    np.isclose(
        final["nivel_nominal"],
        0.95
    ).all()
)

check(
    "Calibracion suficiente para intervalos",
    valid["n_calibracion"].ge(19).all()
)

check(
    "Cobertura empirica entre cero y uno",
    final[
        "cobertura_intervalo_test"
    ].dropna().between(0, 1).all()
)

check(
    "Pronosticos sin duplicados",
    not final.duplicated([
        "barrio",
        "frecuencia",
        "objetivo"
    ]).any()
)

check(
    "Origen anterior al objetivo en test",
    (
        pd.to_datetime(test["origen"])
        < pd.to_datetime(test["fecha"])
    ).all()
)

check(
    "Origen anterior al objetivo en calibracion",
    (
        pd.to_datetime(cal["origen"])
        < pd.to_datetime(cal["fecha"])
    ).all()
)

print("\n" + "=" * 72)
print("PRUEBAS SUPERADAS:", passed)
print("INTERVALOS VALIDADOS")
print("=" * 72)

columns = [
    "barrio",
    "frecuencia",
    "modelo",
    "prediccion",
    "inferior_95",
    "superior_95",
    "WAPE_test",
    "cobertura_intervalo_test"
]

print("\nPRONOSTICOS PRINCIPALES")

print(
    final.loc[
        final["barrio"].isin([
            "BENIFERRI",
            "RUSSAFA",
            "CABANYAL-CANYAMERAR",
            "CABANYAL-CANYAMELAR",
            "TODOS"
        ]),
        columns
    ].round(3).to_string(index=False)
)
