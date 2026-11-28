from pathlib import Path
import numpy as np
import pandas as pd

from src.forecasting import load_series
from src.uncertainty import predict_with_uncertainty
from src.tourism import estimate_demand

target = pd.Timestamp("2026-11-02")

market = pd.read_parquet(
    "app_data/market_context.parquet"
)

zones = sorted(market["barrio"].unique())

results = []
excluded = []

for barrio in zones:

    try:

        y = load_series(barrio, "diaria")

        result, _ = predict_with_uncertainty(
            y, target, "diaria"
        )

        assert result["prediccion"] >= 0
        assert result["n_test"] == 92
        assert result["n_calibracion"] == 182

        assert (
            result["inferior_95"]
            <= result["prediccion"]
            <= result["superior_95"]
        )

        demand = estimate_demand(
            result["prediccion"],
            review_rate=0.5,
            guests=2
        )

        results.append({
            "barrio": barrio,
            **result,
            **demand
        })

    except ValueError as exc:

        excluded.append({
            "barrio": barrio,
            "frecuencia": "diaria",
            "motivo": str(exc)
        })

df = pd.DataFrame(results)

assert not df["barrio"].duplicated().any()

assert len(df) + len(excluded) == len(zones)

assert len(df[df["barrio"] == "TODOS"]) == 1

regional = df[df["barrio"] != "TODOS"]

assert len(regional) == 79

out = Path("results/tables")

df.to_csv(
    out / "daily_final_predictions.csv",
    index=False
)

pd.DataFrame(excluded).to_csv(
    out / "daily_skipped_scopes.csv",
    index=False
)

print("=" * 65)
print("AUDITORIA TERRITORIAL DIARIA")
print("=" * 65)

print("Barrios con resultados:", len(regional))
print("Pronosticos incluyendo ciudad:", len(df))
print("Exclusiones:", len(excluded))

print("\nMODELOS SELECCIONADOS")
print(regional["modelo"].value_counts())

print("\nMEDIANAS DE EVALUACION")
print(regional[[
    "MAE_test",
    "RMSE_test",
    "WAPE_test",
    "cobertura_intervalo_test"
]].median())

print("\nTOP 10 BARRIOS")
print(
    regional.sort_values(
        "prediccion", ascending=False
    )[[
        "barrio",
        "modelo",
        "prediccion",
        "llegadas_estimadas",
        "WAPE_test"
    ]].head(10).round(2).to_string(index=False)
)

print("\nBARRIOS EXCLUIDOS")
print(pd.DataFrame(excluded).to_string(index=False))

print("\nAUDITORIA DIARIA COMPLETADA")
