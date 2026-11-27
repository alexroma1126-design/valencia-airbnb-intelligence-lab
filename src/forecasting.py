from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "app_data"

CONFIG = {
    "semanal": {
        "file": "reviews_semanales.parquet",
        "freq": "W-MON",
        "season": 52
    },
    "mensual": {
        "file": "reviews_mensuales.parquet",
        "freq": "MS",
        "season": 12
    }
}


def load_series(barrio, frequency):

    if frequency not in CONFIG:
        raise ValueError("Frecuencia no reconocida")

    cfg = CONFIG[frequency]

    df = pd.read_parquet(
        DATA / cfg["file"]
    )

    selected = df[
        df["barrio"] == barrio
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No hay datos para {barrio}"
        )

    y = (
        selected.set_index("fecha")["resenas"]
        .sort_index()
        .astype(float)
    )

    y = y.asfreq(cfg["freq"])

    if y.isna().any():
        raise ValueError("Serie temporal incompleta")

    if (y < 0).any():
        raise ValueError("Conteos negativos")

    return y


def forecast_at(
    y,
    target,
    frequency,
    origin=None,
    k=None
):

    cfg = CONFIG[frequency]

    season = cfg["season"]
    freq = cfg["freq"]

    if k is None:
        k = season

    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("K debe ser un entero positivo")

    y = y.sort_index().astype(float)

    if origin is None:
        origin = y.index.max()

    origin = pd.Timestamp(origin)
    target = pd.Timestamp(target)

    if origin not in y.index:
        raise ValueError(
            "El origen no existe en la serie"
        )

    if target <= origin:
        raise ValueError(
            "La fecha objetivo debe ser futura"
        )

    # Solo datos conocidos hasta el origen.
    history = y.loc[:origin].copy()

    future = pd.date_range(
        start=origin,
        end=target,
        freq=freq
    )

    if future[-1] != target:
        raise ValueError(
            "Fecha objetivo incompatible con frecuencia"
        )

    h = len(future) - 1

    # Las reglas actuales requieren que la base
    # estacional sea conocida en el origen.
    if h > season:
        raise ValueError(
            "Horizonte superior al ciclo estacional"
        )

    if frequency == "semanal":

        reference = target - pd.Timedelta(
            weeks=season
        )

    else:

        reference = target - pd.DateOffset(
            months=season
        )

    if reference not in history.index:
        raise ValueError(
            "No existe referencia estacional"
        )

    base = float(history.loc[reference])

    prediction_a = base

    growth = np.nan
    prediction_b = np.nan

    recent_total = np.nan
    previous_total = np.nan

    if len(history) >= season + k:

        recent = history.iloc[-k:]

        previous = history.iloc[
            -season-k:-season
        ]

        recent_total = float(recent.sum())
        previous_total = float(previous.sum())

        if previous_total > 0:

            growth = recent_total / previous_total

            prediction_b = max(
                0.0,
                base * growth
            )

    return {
        "origen": origin,
        "objetivo": target,
        "frecuencia": frequency,
        "horizonte": h,
        "estacionalidad": season,
        "ventana_k": k,
        "fecha_referencia": reference,
        "base_estacional": base,
        "volumen_reciente": recent_total,
        "volumen_anterior": previous_total,
        "factor_crecimiento": growth,
        "modelo_A": prediction_a,
        "modelo_B": prediction_b
    }


def forecast_path(
    y,
    frequency,
    periods=12,
    origin=None,
    k=None
):

    cfg = CONFIG[frequency]

    if origin is None:
        origin = y.index.max()

    origin = pd.Timestamp(origin)

    if periods < 1 or periods > cfg["season"]:
        raise ValueError(
            "Horizonte fuera del rango permitido"
        )

    dates = pd.date_range(
        start=origin,
        periods=periods + 1,
        freq=cfg["freq"]
    )[1:]

    predictions = []

    for target in dates:

        result = forecast_at(
            y=y,
            target=target,
            frequency=frequency,
            origin=origin,
            k=k
        )

        predictions.append(result)

    return pd.DataFrame(predictions)


def demo():

    barrios = [
        "RUSSAFA",
        "CABANYAL-CANYAMELAR",
        "BENIFERRI",
        "TODOS"
    ]

    targets = {
        "semanal": "2026-11-02",
        "mensual": "2026-11-01"
    }

    rows = []

    print("=" * 70)
    print("VALENCIA AIRBNB INTELLIGENCE LAB 3.0")
    print("PRIMEROS PRONOSTICOS ESTACIONALES")
    print("=" * 70)

    for frequency, target in targets.items():

        print(f"\nFRECUENCIA: {frequency.upper()}")

        for barrio in barrios:

            y = load_series(
                barrio,
                frequency
            )

            result = forecast_at(
                y=y,
                target=target,
                frequency=frequency
            )

            result["barrio"] = barrio
            rows.append(result)

            print("\n" + "-" * 60)
            print("BARRIO:", barrio)
            print("Origen:", result["origen"].date())
            print("Objetivo:", result["objetivo"].date())
            print("Horizonte:", result["horizonte"])
            print("Referencia:", result["fecha_referencia"].date())
            print("Base estacional:", result["base_estacional"])
            print("Factor g:", round(result["factor_crecimiento"], 4))
            print("Modelo A:", round(result["modelo_A"], 2))
            print("Modelo B:", round(result["modelo_B"], 2))

    output = ROOT / "results" / "tables"
    output.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(
        output / "forecast_initial_predictions.csv",
        index=False
    )

    print("\nPRONOSTICOS GENERADOS CORRECTAMENTE")


if __name__ == "__main__":
    demo()
