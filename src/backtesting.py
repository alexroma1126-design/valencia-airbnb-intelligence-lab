from pathlib import Path
import numpy as np
import pandas as pd

from src.forecasting import (
    CONFIG,
    load_series,
    forecast_at
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tables"

OUT.mkdir(parents=True, exist_ok=True)

SETTINGS = {
    "diaria": {
        "h": 52,
        "k_values": [7, 28, 91, 364],
        "validation_start": "2024-09-01",
        "validation_end": "2024-11-30",
        "test_start": "2025-10-01",
        "test_end": "2025-12-31"
    },
    "semanal": {
        "h": 10,
        "k_values": [4, 13, 26, 52],
        "validation_start": "2024-09-30",
        "validation_end": "2025-09-22",
        "test_start": "2025-09-29",
        "test_end": "2025-12-29"
    },
    "mensual": {
        "h": 3,
        "k_values": [3, 6, 12],
        "validation_start": "2024-07-01",
        "validation_end": "2025-06-01",
        "test_start": "2025-07-01",
        "test_end": "2025-12-01"
    }
}


def calculate_metrics(real, predicted):

    df = pd.DataFrame({
        "real": real,
        "predicted": predicted
    }).dropna()

    if df.empty:
        return {
            "n": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "WAPE": np.nan,
            "MAPE": np.nan,
            "sesgo": np.nan
        }

    y = df["real"].to_numpy(dtype=float)
    p = df["predicted"].to_numpy(dtype=float)

    error = p - y

    total = y.sum()
    positive = y > 0

    return {
        "n": len(y),
        "MAE": np.mean(np.abs(error)),
        "RMSE": np.sqrt(np.mean(error ** 2)),
        "WAPE": (
            100 * np.abs(error).sum() / total
            if total > 0 else np.nan
        ),
        "MAPE": (
            100 * np.mean(
                np.abs(error[positive]) / y[positive]
            )
            if positive.any() else np.nan
        ),
        "sesgo": np.mean(error)
    }


def get_origin(target, frequency, horizon):

    target = pd.Timestamp(target)

    if frequency == "diaria":

        return target - pd.Timedelta(
            days=horizon
        )

    if frequency == "semanal":

        return target - pd.Timedelta(
            weeks=horizon
        )

    return target - pd.DateOffset(
        months=horizon
    )



def backtest_daily(y, start, end, horizon, k):

    season = CONFIG["diaria"]["season"]

    if not 1 <= horizon <= season:

        raise ValueError("Horizonte diario invalido")

    if k < 1:

        raise ValueError("Ventana K invalida")

    y = y.sort_index().astype(float)

    targets = pd.date_range(
        start=start,
        end=end,
        freq="D"
    )

    origins = targets - pd.Timedelta(
        days=horizon
    )

    references = targets - pd.Timedelta(
        days=season
    )

    mask = (
        targets.isin(y.index)
        & origins.isin(y.index)
        & references.isin(y.index)
    )

    targets = targets[mask]
    origins = origins[mask]
    references = references[mask]

    real = y.reindex(targets).to_numpy()

    base = y.reindex(references).to_numpy()

    rolling = y.rolling(
        window=k,
        min_periods=k
    ).sum()

    recent = rolling.reindex(
        origins
    ).to_numpy()

    previous = rolling.reindex(
        origins - pd.Timedelta(days=season)
    ).to_numpy()

    growth = np.full(
        len(targets),
        np.nan,
        dtype=float
    )

    valid = (
        np.isfinite(recent)
        & np.isfinite(previous)
        & (previous > 0)
    )

    np.divide(
        recent,
        previous,
        out=growth,
        where=valid
    )

    predicted_b = np.maximum(
        0.0,
        base * growth
    )

    return pd.DataFrame({
        "fecha": targets,
        "origen": origins,
        "real": real,
        "modelo_A": base,
        "modelo_B": predicted_b,
        "factor_crecimiento": growth,
        "horizonte": horizon,
        "K": k
    })


def backtest(
    y,
    frequency,
    start,
    end,
    horizon,
    k
):

    if frequency == "diaria":

        return backtest_daily(
            y, start, end, horizon, k
        )

    targets = pd.date_range(
        start=start,
        end=end,
        freq=CONFIG[frequency]["freq"]
    )

    rows = []

    for target in targets:

        origin = get_origin(
            target,
            frequency,
            horizon
        )

        if target not in y.index:
            continue

        if origin not in y.index:
            continue

        try:

            result = forecast_at(
                y=y,
                target=target,
                frequency=frequency,
                origin=origin,
                k=k
            )

        except ValueError:

            continue

        rows.append({
            "fecha": target,
            "origen": origin,
            "real": float(y.loc[target]),
            "modelo_A": result["modelo_A"],
            "modelo_B": result["modelo_B"],
            "factor_crecimiento": (
                result["factor_crecimiento"]
            ),
            "horizonte": horizon,
            "K": k
        })

    return pd.DataFrame(
        rows,
        columns=[
            "fecha",
            "origen",
            "real",
            "modelo_A",
            "modelo_B",
            "factor_crecimiento",
            "horizonte",
            "K"
        ]
    )


def evaluate_candidates(
    y,
    frequency,
    start,
    end,
    horizon,
    k_values
):

    candidates = []
    predictions = {}

    base = backtest(
        y,
        frequency,
        start,
        end,
        horizon,
        k_values[0]
    )

    predictions["A"] = base

    for k in k_values:

        predictions[f"B_K{k}"] = backtest(
            y,
            frequency,
            start,
            end,
            horizon,
            k
        )

    expected = len(pd.date_range(
        start,
        end,
        freq=CONFIG[frequency]["freq"]
    ))

    for name, frame in predictions.items():

        column = (
            "modelo_A"
            if name == "A"
            else "modelo_B"
        )

        metrics = calculate_metrics(
            frame["real"],
            frame[column]
        )

        k = (
            0
            if name == "A"
            else int(name.split("K")[-1])
        )

        candidates.append({
            "modelo": name,
            "K": k,
            "periodos_esperados": expected,
            "cobertura": (
                metrics["n"] / expected
            ),
            **metrics
        })

    return pd.DataFrame(
        candidates
    ).sort_values(
        ["MAE", "RMSE", "modelo"]
    )


def select_model(validation):

    valid = validation[
        (validation["cobertura"] == 1)
        & validation["MAE"].notna()
    ]

    if valid.empty:
        return None

    return valid.sort_values(
        ["MAE", "RMSE", "modelo"]
    ).iloc[0]


def evaluate_scope(barrio, frequency):

    settings = SETTINGS[frequency]

    y = load_series(
        barrio,
        frequency
    )

    validation = evaluate_candidates(
        y=y,
        frequency=frequency,
        start=settings["validation_start"],
        end=settings["validation_end"],
        horizon=settings["h"],
        k_values=settings["k_values"]
    )

    winner = select_model(validation)

    validation["barrio"] = barrio
    validation["frecuencia"] = frequency

    if winner is None:

        return validation, None, None

    model = winner["modelo"]

    k = (
        int(winner["K"])
        if model != "A"
        else CONFIG[frequency]["season"]
    )

    test = backtest(
        y=y,
        frequency=frequency,
        start=settings["test_start"],
        end=settings["test_end"],
        horizon=settings["h"],
        k=k
    )

    column = (
        "modelo_A"
        if model == "A"
        else "modelo_B"
    )

    test["prediccion"] = test[column]

    test["error"] = (
        test["prediccion"] - test["real"]
    )

    test["error_absoluto"] = (
        test["error"].abs()
    )

    test["barrio"] = barrio
    test["frecuencia"] = frequency
    test["modelo_seleccionado"] = model

    metrics = calculate_metrics(
        test["real"],
        test["prediccion"]
    )

    baseline = calculate_metrics(
        test["real"],
        test["modelo_A"]
    )

    expected = len(pd.date_range(
        settings["test_start"],
        settings["test_end"],
        freq=CONFIG[frequency]["freq"]
    ))

    result = {
        "barrio": barrio,
        "frecuencia": frequency,
        "modelo_seleccionado": model,
        "K": k,
        "horizonte": settings["h"],
        "MAE_validacion": winner["MAE"],
        "test_esperado": expected,
        "cobertura_test": metrics["n"] / expected,
        "MAE_baseline": baseline["MAE"],
        "mejora_MAE_pct": (
            100 * (
                baseline["MAE"] - metrics["MAE"]
            ) / baseline["MAE"]
            if baseline["MAE"] > 0
            else np.nan
        ),
        **metrics
    }

    return validation, result, test


def main():

    print("=" * 72)
    print("VALENCIA AIRBNB INTELLIGENCE LAB 3.0")
    print("VALIDACION TEMPORAL Y BACKTESTING")
    print("=" * 72)

    validation_rows = []
    test_rows = []
    prediction_rows = []
    skipped = []

    for frequency, cfg in CONFIG.items():

        df = pd.read_parquet(
            ROOT / "app_data" / cfg["file"]
        )

        barrios = sorted(
            df["barrio"].unique()
        )

        print(
            f"\n{frequency.upper()}: "
            f"{len(barrios)} ambitos"
        )

        for barrio in barrios:

            validation, result, predictions = (
                evaluate_scope(
                    barrio,
                    frequency
                )
            )

            validation_rows.append(validation)

            if result is None:

                skipped.append({
                    "barrio": barrio,
                    "frecuencia": frequency,
                    "motivo": (
                        "Validacion insuficiente"
                    )
                })

                continue

            test_rows.append(result)
            prediction_rows.append(predictions)

            if barrio in [
                "RUSSAFA",
                "CABANYAL-CANYAMELAR",
                "BENIFERRI",
                "TODOS"
            ]:

                print("\nBARRIO:", barrio)
                print(
                    "Modelo elegido:",
                    result["modelo_seleccionado"]
                )
                print(
                    "MAE validacion:",
                    round(
                        result["MAE_validacion"], 3
                    )
                )
                print(
                    "MAE test:",
                    round(result["MAE"], 3)
                )
                print(
                    "RMSE test:",
                    round(result["RMSE"], 3)
                )
                print(
                    "WAPE test:",
                    round(result["WAPE"], 3)
                )
                print(
                    "Mejora MAE vs A:",
                    round(
                        result["mejora_MAE_pct"], 3
                    )
                )

    validation_df = pd.concat(
        validation_rows,
        ignore_index=True
    )

    test_df = pd.DataFrame(test_rows)

    predictions_df = pd.concat(
        prediction_rows,
        ignore_index=True
    )

    validation_df.to_csv(
        OUT / "forecast_validation_metrics.csv",
        index=False
    )

    test_df.to_csv(
        OUT / "forecast_test_metrics.csv",
        index=False
    )

    predictions_df.to_csv(
        OUT / "forecast_test_predictions.csv",
        index=False
    )

    pd.DataFrame(
        skipped,
        columns=[
            "barrio",
            "frecuencia",
            "motivo"
        ]
    ).to_csv(
        OUT / "forecast_skipped_scopes.csv",
        index=False
    )

    print("\n" + "=" * 72)
    print("RESUMEN GLOBAL")
    print("=" * 72)

    print(
        "\nAmbitos evaluados:",
        len(test_df)
    )

    print(
        "Ambitos sin validacion:",
        len(skipped)
    )

    print("\nMODELOS SELECCIONADOS")

    print(
        test_df.groupby([
            "frecuencia",
            "modelo_seleccionado"
        ]).size().to_string()
    )

    print("\nRESUMEN DE COBERTURA")

    print(
        test_df.groupby("frecuencia")[
            "cobertura_test"
        ].describe().round(3).to_string()
    )

    print("\nBACKTESTING COMPLETADO")


if __name__ == "__main__":
    main()
