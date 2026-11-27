import numpy as np
import pandas as pd

from src.forecasting import (
    load_series,
    forecast_at,
    forecast_path
)

passed = 0


def check(name, condition):

    global passed

    if not condition:
        raise AssertionError(name)

    passed += 1
    print(f"[OK] {name}")


print("=" * 65)
print("VALIDACION MATEMATICA DEL MOTOR PREDICTIVO")
print("=" * 65)

# PRUEBA 1: Serie constante semanal

index = pd.date_range(
    "2020-01-06",
    periods=156,
    freq="W-MON"
)

y = pd.Series(
    20.0,
    index=index
)

target = index[-1] + pd.Timedelta(weeks=10)

r = forecast_at(
    y,
    target,
    "semanal"
)

check(
    "Serie constante: modelo A = 20",
    np.isclose(r["modelo_A"], 20)
)

check(
    "Serie constante: modelo B = 20",
    np.isclose(r["modelo_B"], 20)
)

check(
    "Serie constante: crecimiento = 1",
    np.isclose(r["factor_crecimiento"], 1)
)

check(
    "Horizonte semanal correcto",
    r["horizonte"] == 10
)

# PRUEBA 2: Crecimiento interanual conocido

index = pd.date_range(
    "2020-01-01",
    periods=24,
    freq="MS"
)

y = pd.Series(
    [20.0] * 12 + [30.0] * 12,
    index=index
)

r = forecast_at(
    y,
    "2022-01-01",
    "mensual"
)

check(
    "Crecimiento interanual de 50 por ciento",
    np.isclose(r["factor_crecimiento"], 1.5)
)

check(
    "Modelo A mensual = 30",
    np.isclose(r["modelo_A"], 30)
)

check(
    "Modelo B mensual = 45",
    np.isclose(r["modelo_B"], 45)
)

# PRUEBA 3: Ausencia de fuga temporal

y = load_series(
    "RUSSAFA",
    "semanal"
)

origin = y.index[-25]
target = origin + pd.Timedelta(weeks=10)

r1 = forecast_at(
    y,
    target,
    "semanal",
    origin=origin
)

modified = y.copy()

modified.loc[
    modified.index > origin
] = 999999

r2 = forecast_at(
    modified,
    target,
    "semanal",
    origin=origin
)

check(
    "Modelo A no utiliza datos futuros",
    np.isclose(r1["modelo_A"], r2["modelo_A"])
)

check(
    "Modelo B no utiliza datos futuros",
    np.isclose(r1["modelo_B"], r2["modelo_B"])
)

# PRUEBA 4: Denominador cero

index = pd.date_range(
    "2020-01-01",
    periods=24,
    freq="MS"
)

y = pd.Series(
    [0.0] * 12 + [10.0] * 12,
    index=index
)

r = forecast_at(
    y,
    "2022-01-01",
    "mensual"
)

check(
    "Division por cero controlada",
    np.isnan(r["modelo_B"])
)

check(
    "Modelo A disponible cuando B no existe",
    np.isclose(r["modelo_A"], 10)
)

# PRUEBA 5: Pronostico de varios periodos

y = load_series(
    "CABANYAL-CANYAMELAR",
    "mensual"
)

path = forecast_path(
    y,
    "mensual",
    periods=12
)

check(
    "Pronostico de 12 meses",
    len(path) == 12
)

check(
    "Predicciones A no negativas",
    path["modelo_A"].ge(0).all()
)

check(
    "Predicciones B no negativas",
    path["modelo_B"].ge(0).all()
)

check(
    "Fechas objetivo posteriores al origen",
    (path["objetivo"] > path["origen"]).all()
)

check(
    "Origen fijo durante todo el pronostico",
    path["origen"].nunique() == 1
)

# PRUEBA 6: Horizonte superior al permitido

try:

    forecast_path(
        y,
        "mensual",
        periods=13
    )

except ValueError:

    rejected = True

else:

    rejected = False

check(
    "Horizonte excesivo rechazado",
    rejected
)

print("\n" + "=" * 65)
print("PRUEBAS SUPERADAS:", passed)
print("MOTOR PREDICTIVO VALIDADO")
print("=" * 65)
