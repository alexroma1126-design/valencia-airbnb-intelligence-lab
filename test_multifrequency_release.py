import pandas as pd
import numpy as np

from src.forecasting import load_series
from src.uncertainty import predict_with_uncertainty

daily = pd.read_csv("results/tables/daily_final_predictions.csv")
excluded = pd.read_csv("results/tables/daily_skipped_scopes.csv")
previous = pd.read_csv("results/tables/forecast_final_predictions.csv")

assert len(daily) == 80
assert len(excluded) == 5
assert len(previous) == 144
assert not daily["barrio"].duplicated().any()
assert daily["frecuencia"].eq("diaria").all()
assert daily["n_test"].eq(92).all()
assert daily["n_calibracion"].eq(182).all()
assert daily["cobertura_datos_test"].eq(1).all()
assert daily["prediccion"].ge(0).all()
assert (daily["inferior_95"] <= daily["prediccion"]).all()
assert (daily["prediccion"] <= daily["superior_95"]).all()
assert not set(daily["barrio"]).intersection(excluded["barrio"])

print("[OK] 80 pronosticos diarios")
print("[OK] 5 exclusiones justificadas")
print("[OK] Intervalos consistentes")
print("[OK] Calibracion y test completos")
print("[OK] 144 pronosticos anteriores conservados")

for i, row in enumerate(daily.itertuples(index=False), 1):

    summary, detail = predict_with_uncertainty(
        load_series(row.barrio, "diaria"),
        row.objetivo,
        "diaria"
    )

    assert np.isclose(row.prediccion, summary["prediccion"])
    assert row.modelo == summary["modelo"]
    assert np.isclose(row.MAE_test, summary["MAE_test"])
    assert row.inferior_95 == summary["inferior_95"]
    assert row.superior_95 == summary["superior_95"]

    test = detail["test"]

    assert len(test) == 92
    assert (test["origen"] < test["fecha"]).all()
    assert (test["fecha"] - test["origen"]).dt.days.eq(52).all()
    assert len(detail["calibracion"]) == 182

    print(f"[OK] {i}/80 - {row.barrio}")

print("\nTOTAL PRONOSTICOS:", len(daily) + len(previous))
print("\nBARRIOS SIN PRONOSTICO DIARIO:")
print(excluded.to_string(index=False))
print("\nVALIDACION MULTIFRECUENCIA COMPLETADA")
