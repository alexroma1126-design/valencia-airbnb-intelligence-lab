from pathlib import Path
import numpy as np
import pandas as pd

from src.forecasting import (
    CONFIG,
    load_series,
    forecast_at
)

from src.backtesting import (
    backtest,
    SETTINGS,
    evaluate_candidates
)

from src.uncertainty import (
    predict_with_uncertainty,
    temporal_windows
)

from src.tourism import estimate_demand


print("=" * 65)
print("VALIDACION CIENTIFICA DIARIA")
print("=" * 65)


df = pd.read_parquet(
    "app_data/reviews_diarias.parquet"
)

assert not df.duplicated(
    ["barrio", "fecha"]
).any()

assert df["resenas"].ge(0).all()

print("[OK] Integridad de la base diaria")


y = load_series("RUSSAFA", "diaria")

origin = y.index.max()

target = pd.Timestamp("2026-11-02")

result = forecast_at(
    y,
    target,
    "diaria",
    k=28
)

reference = target - pd.Timedelta(days=364)

assert result["fecha_referencia"] == reference

assert result["modelo_A"] == y.loc[reference]

assert reference.day_of_week == target.day_of_week

print("[OK] Estacionalidad de 364 dias")


# Check the vectorized engine against the original formula.

bt = backtest(
    y,
    "diaria",
    "2025-10-01",
    "2025-10-10",
    52,
    28
)

assert len(bt) == 10

for row in bt.itertuples(index=False):

    direct = forecast_at(
        y,
        row.fecha,
        "diaria",
        origin=row.origen,
        k=28
    )

    assert np.isclose(
        row.modelo_A,
        direct["modelo_A"]
    )

    assert np.isclose(
        row.modelo_B,
        direct["modelo_B"],
        equal_nan=True
    )

print("[OK] Backtesting coincide con formula original")


# Ensure future observations do not affect predictions.

modified = y.copy()

modified.loc[modified.index > bt.iloc[0]["origen"]] += 10000

first = bt.iloc[0]

a = forecast_at(
    y,
    first["fecha"],
    "diaria",
    origin=first["origen"],
    k=28
)

b = forecast_at(
    modified,
    first["fecha"],
    "diaria",
    origin=first["origen"],
    k=28
)

assert np.isclose(
    a["modelo_B"],
    b["modelo_B"],
    equal_nan=True
)

print("[OK] Sin fuga de informacion futura")


horizon = (target - origin).days

windows = temporal_windows(
    "diaria",
    horizon
)

assert (
    windows["validation_end"]
    <= windows["calibration_start"]
    - pd.Timedelta(days=horizon)
)

assert (
    windows["calibration_end"]
    <= windows["first_test_origin"]
)

print("[OK] Separacion cronologica correcta")


results = []
failures = []

for barrio in [
    "RUSSAFA",
    "CABANYAL-CANYAMELAR",
    "BENIFERRI",
    "TODOS"
]:

    try:

        series = load_series(
            barrio,
            "diaria"
        )

        summary, detail = predict_with_uncertainty(
            series,
            target,
            "diaria"
        )

        assert summary["prediccion"] >= 0

        assert (
            summary["inferior_95"]
            <= summary["prediccion"]
            <= summary["superior_95"]
        )

        assert summary["n_calibracion"] == 182

        assert summary["n_test"] == 92

        assert summary["cobertura_datos_test"] == 1

        demand = estimate_demand(
            summary["prediccion"],
            review_rate=0.5,
            guests=2
        )

        row = {
            "barrio": barrio,
            **summary,
            **demand
        }

        results.append(row)

        print("\nBARRIO:", barrio)
        print("Modelo:", summary["modelo"])
        print("Horizonte:", summary["horizonte"])
        print("Resenas:", round(summary["prediccion"], 2))
        print("Estancias:", round(demand["estancias_estimadas"], 2))
        print("Llegadas:", round(demand["llegadas_estimadas"], 2))
        print(
            "Intervalo nominal:",
            summary["inferior_95"],
            "-",
            summary["superior_95"]
        )
        print("MAE:", round(summary["MAE_test"], 3))
        print("WAPE:", round(summary["WAPE_test"], 2))
        print(
            "Cobertura:",
            round(
                100 * summary["cobertura_intervalo_test"],
                2
            )
        )

    except ValueError as exc:

        failures.append({
            "barrio": barrio,
            "motivo": str(exc)
        })


assert any(
    r["barrio"] == "TODOS"
    for r in results
)

assert any(
    r["barrio"] != "TODOS"
    for r in results
)

out = Path("results/tables")

out.mkdir(parents=True, exist_ok=True)

pd.DataFrame(results).to_csv(
    out / "daily_validation_examples.csv",
    index=False
)

print("\nAMBITOS VALIDOS:", len(results))
print("AMBITOS EXCLUIDOS:", failures)

print("\nPRUEBAS CIENTIFICAS DIARIAS SUPERADAS")
