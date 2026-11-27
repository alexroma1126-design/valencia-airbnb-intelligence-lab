from pathlib import Path
import math
import numpy as np
import pandas as pd

from src.forecasting import (
    CONFIG,
    load_series,
    forecast_at
)

from src.backtesting import (
    SETTINGS,
    backtest,
    evaluate_candidates,
    select_model,
    calculate_metrics,
    get_origin
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tables"

OUT.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05


def shift_period(date, frequency, periods):

    date = pd.Timestamp(date)

    if frequency == "semanal":

        return date + pd.Timedelta(
            weeks=periods
        )

    return date + pd.DateOffset(
        months=periods
    )


def temporal_windows(frequency, horizon):

    settings = SETTINGS[frequency]

    test_start = pd.Timestamp(
        settings["test_start"]
    )

    test_end = pd.Timestamp(
        settings["test_end"]
    )

    n_cal = (
        78 if frequency == "semanal"
        else 24
    )

    n_val = (
        52 if frequency == "semanal"
        else 12
    )

    first_test_origin = get_origin(
        test_start,
        frequency,
        horizon
    )

    cal_end = first_test_origin

    cal_start = shift_period(
        cal_end,
        frequency,
        -(n_cal - 1)
    )

    val_end = get_origin(
        cal_start,
        frequency,
        horizon
    )

    val_start = shift_period(
        val_end,
        frequency,
        -(n_val - 1)
    )

    return {
        "validation_start": val_start,
        "validation_end": val_end,
        "calibration_start": cal_start,
        "calibration_end": cal_end,
        "test_start": test_start,
        "test_end": test_end,
        "first_test_origin": first_test_origin
    }


def empirical_radius(errors, alpha=ALPHA):

    errors = np.asarray(
        errors,
        dtype=float
    )

    errors = np.sort(
        np.abs(
            errors[np.isfinite(errors)]
        )
    )

    n = len(errors)

    if n == 0:

        return np.nan

    rank = math.ceil(
        (n + 1) * (1 - alpha)
    )

    if rank > n:

        return np.nan

    return float(
        errors[rank - 1]
    )


def prediction_interval(prediction, radius):

    if not np.isfinite(prediction):

        return np.nan, np.nan

    if not np.isfinite(radius):

        return np.nan, np.nan

    lower = max(
        0,
        math.floor(prediction - radius)
    )

    upper = max(
        lower,
        math.ceil(prediction + radius)
    )

    return lower, upper


def calibrate_model(
    y,
    frequency,
    horizon
):

    windows = temporal_windows(
        frequency,
        horizon
    )

    validation = evaluate_candidates(
        y=y,
        frequency=frequency,
        start=windows["validation_start"],
        end=windows["validation_end"],
        horizon=horizon,
        k_values=SETTINGS[frequency]["k_values"]
    )

    winner = select_model(validation)

    if winner is None:

        return None

    model = winner["modelo"]

    k = (
        CONFIG[frequency]["season"]
        if model == "A"
        else int(winner["K"])
    )

    column = (
        "modelo_A"
        if model == "A"
        else "modelo_B"
    )

    calibration = backtest(
        y=y,
        frequency=frequency,
        start=windows["calibration_start"],
        end=windows["calibration_end"],
        horizon=horizon,
        k=k
    )

    calibration["prediccion"] = (
        calibration[column]
    )

    calibration = calibration.dropna(
        subset=["real", "prediccion"]
    ).copy()

    calibration["error"] = (
        calibration["real"]
        - calibration["prediccion"]
    )

    radius = empirical_radius(
        calibration["error"]
    )

    test = backtest(
        y=y,
        frequency=frequency,
        start=windows["test_start"],
        end=windows["test_end"],
        horizon=horizon,
        k=k
    )

    test["prediccion"] = test[column]

    test = test.dropna(
        subset=["real", "prediccion"]
    ).copy()

    limits = [
        prediction_interval(p, radius)
        for p in test["prediccion"]
    ]

    test["inferior"] = [
        x[0] for x in limits
    ]

    test["superior"] = [
        x[1] for x in limits
    ]

    metrics = calculate_metrics(
        test["real"],
        test["prediccion"]
    )

    baseline = calculate_metrics(
        test["real"],
        test["modelo_A"]
    )

    expected = len(pd.date_range(
        windows["test_start"],
        windows["test_end"],
        freq=CONFIG[frequency]["freq"]
    ))

    if np.isfinite(radius) and len(test):

        test["cubierto"] = (
            (test["real"] >= test["inferior"])
            & (test["real"] <= test["superior"])
        )

        coverage = float(
            test["cubierto"].mean()
        )

    else:

        test["cubierto"] = np.nan
        coverage = np.nan

    return {
        "modelo": model,
        "K": k,
        "horizonte": horizon,
        "radio": radius,
        "n_calibracion": len(calibration),
        "MAE_validacion": float(winner["MAE"]),
        "MAE_test": metrics["MAE"],
        "RMSE_test": metrics["RMSE"],
        "WAPE_test": metrics["WAPE"],
        "MAE_baseline": baseline["MAE"],
        "n_test": metrics["n"],
        "cobertura_datos_test": (
            metrics["n"] / expected
        ),
        "cobertura_intervalo": coverage,
        "ventanas": windows,
        "test": test,
        "calibracion": calibration
    }


def predict_with_uncertainty(
    y,
    target,
    frequency,
    origin=None
):

    if origin is None:

        origin = y.index.max()

    origin = pd.Timestamp(origin)
    target = pd.Timestamp(target)

    history = y.loc[:origin].copy()

    if origin not in history.index:

        raise ValueError(
            "Origen no disponible"
        )

    dates = pd.date_range(
        origin,
        target,
        freq=CONFIG[frequency]["freq"]
    )

    if len(dates) < 2 or dates[-1] != target:

        raise ValueError(
            "Fecha objetivo incorrecta"
        )

    horizon = len(dates) - 1

    if horizon > CONFIG[frequency]["season"]:

        raise ValueError(
            "Horizonte superior al ciclo estacional"
        )

    calibrated = calibrate_model(
        history,
        frequency,
        horizon
    )

    if calibrated is None:

        raise ValueError(
            "Historia insuficiente para seleccionar modelo"
        )

    test = calibrated["test"]

    expected_test_end = (
        calibrated["ventanas"]["test_end"]
    )

    if origin < expected_test_end:

        raise ValueError(
            "Origen anterior al cierre del test"
        )

    forecast = forecast_at(
        y=history,
        target=target,
        frequency=frequency,
        origin=origin,
        k=calibrated["K"]
    )

    model = calibrated["modelo"]

    value = (
        forecast["modelo_A"]
        if model == "A"
        else forecast["modelo_B"]
    )

    if not np.isfinite(value):

        raise ValueError(
            "Modelo seleccionado no disponible"
        )

    lower, upper = prediction_interval(
        value,
        calibrated["radio"]
    )

    summary = {
        "origen": origin,
        "objetivo": target,
        "frecuencia": frequency,
        "horizonte": horizon,
        "modelo": model,
        "K": calibrated["K"],
        "prediccion": float(value),
        "inferior_95": lower,
        "superior_95": upper,
        "radio_intervalo": calibrated["radio"],
        "nivel_nominal": 0.95,
        "n_calibracion": (
            calibrated["n_calibracion"]
        ),
        "n_test": calibrated["n_test"],
        "MAE_validacion": (
            calibrated["MAE_validacion"]
        ),
        "MAE_test": calibrated["MAE_test"],
        "RMSE_test": calibrated["RMSE_test"],
        "WAPE_test": calibrated["WAPE_test"],
        "MAE_baseline": (
            calibrated["MAE_baseline"]
        ),
        "cobertura_datos_test": (
            calibrated["cobertura_datos_test"]
        ),
        "cobertura_intervalo_test": (
            calibrated["cobertura_intervalo"]
        )
    }

    return summary, calibrated


def main():

    print("=" * 72)
    print("VALENCIA AIRBNB INTELLIGENCE LAB 3.0")
    print("PRONOSTICOS E INTERVALOS DE PREDICCION")
    print("=" * 72)

    targets = {
        "semanal": "2026-11-02",
        "mensual": "2026-11-01"
    }

    results = []
    tests = []
    calibrations = []
    skipped = []

    for frequency, target in targets.items():

        cfg = CONFIG[frequency]

        data = pd.read_parquet(
            ROOT / "app_data" / cfg["file"]
        )

        barrios = sorted(
            data["barrio"].unique()
        )

        print(
            "\nPROCESANDO:",
            frequency.upper()
        )

        for barrio in barrios:

            y = load_series(
                barrio,
                frequency
            )

            try:

                result, detail = (
                    predict_with_uncertainty(
                        y,
                        target,
                        frequency
                    )
                )

            except ValueError as exc:

                skipped.append({
                    "barrio": barrio,
                    "frecuencia": frequency,
                    "motivo": str(exc)
                })

                continue

            result["barrio"] = barrio
            results.append(result)

            test = detail["test"].copy()

            test["barrio"] = barrio
            test["frecuencia"] = frequency
            test["modelo"] = result["modelo"]

            tests.append(test)

            cal = detail["calibracion"].copy()

            cal["barrio"] = barrio
            cal["frecuencia"] = frequency
            cal["modelo"] = result["modelo"]

            calibrations.append(cal)

            if barrio in [
                "BENIFERRI",
                "RUSSAFA",
                "CABANYAL-CANYAMELAR",
                "TODOS"
            ]:

                print("\nBARRIO:", barrio)

                print(
                    "Modelo:",
                    result["modelo"]
                )

                print(
                    "Horizonte:",
                    result["horizonte"]
                )

                print(
                    "Prediccion:",
                    round(
                        result["prediccion"],
                        2
                    )
                )

                print(
                    "Intervalo nominal 95%:",
                    result["inferior_95"],
                    "-",
                    result["superior_95"]
                )

                print(
                    "MAE test:",
                    round(result["MAE_test"], 3)
                )

                print(
                    "WAPE test:",
                    round(result["WAPE_test"], 3)
                )

                print(
                    "Observaciones calibracion:",
                    result["n_calibracion"]
                )

                print(
                    "Cobertura intervalo test:",
                    round(
                        100 * result[
                            "cobertura_intervalo_test"
                        ],
                        2
                    )
                )

    final = pd.DataFrame(results)

    if final.empty:

        raise RuntimeError(
            "No se generaron pronosticos"
        )

    final.to_csv(
        OUT / "forecast_final_predictions.csv",
        index=False
    )

    pd.concat(
        tests,
        ignore_index=True
    ).to_csv(
        OUT / "forecast_interval_test.csv",
        index=False
    )

    pd.concat(
        calibrations,
        ignore_index=True
    ).to_csv(
        OUT / "forecast_calibration.csv",
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
        OUT / "forecast_uncertainty_skipped.csv",
        index=False
    )

    print("\n" + "=" * 72)
    print("RESUMEN GLOBAL")
    print("=" * 72)

    print(
        "Pronosticos generados:",
        len(final)
    )

    print(
        "Ambitos excluidos:",
        len(skipped)
    )

    print(
        "Intervalos disponibles:",
        final["inferior_95"].notna().sum()
    )

    print("\nRESUMEN POR FRECUENCIA")

    print(
        final.groupby("frecuencia")[
            [
                "MAE_test",
                "WAPE_test",
                "n_calibracion",
                "cobertura_intervalo_test"
            ]
        ].median().round(3).to_string()
    )

    print(
        "\nPRONOSTICOS E INTERVALOS GENERADOS"
    )


if __name__ == "__main__":
    main()
