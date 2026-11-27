from pathlib import Path
import numpy as np
import pandas as pd

from src.tourism import (
    estimate_demand,
    estimate_scenarios,
    period_context
)

ROOT = Path.cwd()
DATA = ROOT / "app_data"
OUT = ROOT / "results" / "tables"

passed = 0


def check(name, condition):

    global passed

    if not condition:
        raise AssertionError(name)

    passed += 1

    print(f"[OK] {name}")


print("=" * 70)
print("VALIDACION DE ESTIMACIONES TURISTICAS")
print("=" * 70)

r = estimate_demand(24, 0.50, 2)

check(
    "Conversion correcta a estancias",
    np.isclose(
        r["estancias_estimadas"], 48
    )
)

check(
    "Conversion correcta a huespedes",
    np.isclose(
        r["llegadas_estimadas"], 96
    )
)

check(
    "Cero resenas produce cero llegadas",
    estimate_demand(0)["llegadas_estimadas"] == 0
)

try:

    estimate_demand(24, 0, 2)

except ValueError:

    rejected = True

else:

    rejected = False

check(
    "Tasa cero rechazada",
    rejected
)

try:

    estimate_demand(24, 1.5, 2)

except ValueError:

    rejected = True

else:

    rejected = False

check(
    "Tasa superior a uno rechazada",
    rejected
)

try:

    estimate_demand(-1)

except ValueError:

    rejected = True

else:

    rejected = False

check(
    "Resenas negativas rechazadas",
    rejected
)

s = estimate_scenarios(
    24,
    19,
    29
)

check(
    "Escenarios ordenados correctamente",
    s["huespedes_escenario_bajo"]
    <= s["huespedes_central"]
    <= s["huespedes_escenario_alto"]
)

market = pd.read_parquet(
    DATA / "market_context.parquet"
)

daily = pd.read_parquet(
    DATA / "calendar_daily_context.parquet"
)

report = pd.read_csv(
    OUT / "forecast_tourism_estimates.csv"
)

final = pd.read_csv(
    OUT / "forecast_final_predictions.csv"
)

check(
    "Existen estimaciones",
    not report.empty
)

check(
    "Todos los pronosticos con intervalo procesados",
    len(report) == final.dropna(
        subset=["inferior_95", "superior_95"]
    ).shape[0]
)

check(
    "Estimaciones sin duplicados",
    not report.duplicated([
        "barrio",
        "frecuencia",
        "objetivo"
    ]).any()
)

check(
    "Capacidad no negativa",
    market["capacidad_total"].ge(0).all()
)

check(
    "Disponibilidad entre cero y cien",
    daily["disponibilidad_pct"]
    .dropna().between(0, 100).all()
)

check(
    "Identidad de disponibilidad",
    np.isclose(
        daily["disponibles"]
        + daily["no_disponibles"],
        daily["registros_validos"]
    ).all()
)

check(
    "Calendario diario sin duplicados",
    not daily.duplicated([
        "barrio",
        "fecha"
    ]).any()
)

check(
    "Estancias estimadas no negativas",
    report["estancias_central"].ge(0).all()
)

check(
    "Llegadas estimadas no negativas",
    report["huespedes_central"].ge(0).all()
)

check(
    "Escenarios contienen valor central",
    (
        (
            report["huespedes_escenario_bajo"]
            <= report["huespedes_central"]
        )
        & (
            report["huespedes_central"]
            <= report["huespedes_escenario_alto"]
        )
    ).all()
)

r = period_context(
    daily,
    "TODOS",
    "2026-11-01",
    "mensual"
)

check(
    "Noviembre contiene 30 dias",
    r["dias_esperados"] == 30
)

check(
    "Calendario de noviembre disponible",
    r["calendario_completo"]
)

r = period_context(
    daily,
    "TODOS",
    "2030-01-01",
    "mensual"
)

check(
    "Calendario fuera de rango controlado",
    not r["calendario_completo"]
    and np.isnan(
        r["disponibilidad_media_pct"]
    )
)

print("\n" + "=" * 70)
print("PRUEBAS SUPERADAS:", passed)
print("ESTIMACIONES TURISTICAS VALIDADAS")
print("=" * 70)
