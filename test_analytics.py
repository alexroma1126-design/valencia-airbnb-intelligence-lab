import pandas as pd
import numpy as np

from src.analytics import (
    descriptive_statistics,
    moving_average,
    temporal_decomposition,
    autocorrelation_analysis,
    available_stay
)

print("=" * 60)
print("PRUEBAS DEL LABORATORIO ANALITICO")
print("=" * 60)

rng = np.random.default_rng(42)

dates = pd.date_range(
    "2021-01-01",
    periods=60,
    freq="MS"
)

t = np.arange(60)

series = pd.Series(
    100 + 2*t
    + 25*np.sin(2*np.pi*t/12)
    + rng.normal(0, 5, 60),
    index=dates
)

print("\nESTADISTICA DESCRIPTIVA")
print(descriptive_statistics(series).round(2))

print("\nMEDIA MOVIL")
print(moving_average(series, 3).tail())

print("\nDESCOMPOSICION STL")
print(temporal_decomposition(series).tail())

print("\nACF Y PACF")
print(autocorrelation_analysis(series).head())

calendar = pd.DataFrame({
    "listing_id": [1, 1, 1, 2, 2, 2],
    "date": list(pd.date_range("2026-12-01", periods=3))*2,
    "available_numeric": [1, 1, 1, 1, 0, 1],
    "minimum_nights": [1]*6,
    "maximum_nights": [30]*6
})

assert available_stay(
    calendar, "2026-12-01", 3
) == [1]

print("\nPRUEBA DE DISPONIBILIDAD: OK")

print("\nTODAS LAS PRUEBAS FINALIZADAS")
