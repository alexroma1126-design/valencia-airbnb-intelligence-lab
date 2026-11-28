from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DATA = ROOT / "app_data"
OUT = ROOT / "results" / "tables"

DATA.mkdir(exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

DEFAULT_RATE = 0.50
DEFAULT_GUESTS = 2.0


def estimate_demand(
    reviews,
    review_rate=DEFAULT_RATE,
    guests=DEFAULT_GUESTS
):

    if not np.isfinite(reviews) or reviews < 0:
        raise ValueError("Resenas invalidas")

    if not 0 < review_rate <= 1:
        raise ValueError("Tasa fuera de rango")

    if not np.isfinite(guests) or guests < 1:
        raise ValueError("Huespedes invalidos")

    stays = reviews / review_rate

    arrivals = stays * guests

    return {
        "resenas": float(reviews),
        "tasa_resena": float(review_rate),
        "huespedes_por_estancia": float(guests),
        "estancias_estimadas": float(stays),
        "llegadas_estimadas": float(arrivals)
    }


def estimate_scenarios(
    reviews,
    lower,
    upper
):

    central = estimate_demand(
        reviews,
        0.50,
        2.0
    )

    low = estimate_demand(
        lower,
        0.70,
        1.5
    )

    high = estimate_demand(
        upper,
        0.30,
        3.0
    )

    return {
        "estancias_central": (
            central["estancias_estimadas"]
        ),
        "estancias_escenario_bajo": (
            low["estancias_estimadas"]
        ),
        "estancias_escenario_alto": (
            high["estancias_estimadas"]
        ),
        "huespedes_central": (
            central["llegadas_estimadas"]
        ),
        "huespedes_escenario_bajo": (
            low["llegadas_estimadas"]
        ),
        "huespedes_escenario_alto": (
            high["llegadas_estimadas"]
        )
    }


def prepare_market_context():

    listings = pd.read_csv(
        RAW / "listings.csv.gz",
        usecols=[
            "id",
            "neighbourhood_cleansed",
            "accommodates"
        ]
    )

    listings = listings.rename(
        columns={
            "id": "listing_id",
            "neighbourhood_cleansed": "barrio"
        }
    )

    listings["accommodates"] = pd.to_numeric(
        listings["accommodates"],
        errors="coerce"
    )

    listings.loc[
        listings["accommodates"] <= 0,
        "accommodates"
    ] = np.nan

    if listings["listing_id"].duplicated().any():
        raise ValueError("IDs duplicados")

    regional = listings.groupby(
        "barrio"
    ).agg(
        alojamientos=("listing_id", "nunique"),
        capacidad_total=("accommodates", "sum"),
        capacidad_media=("accommodates", "mean"),
        capacidad_mediana=("accommodates", "median"),
        capacidad_valida=("accommodates", "count")
    ).reset_index()

    city = pd.DataFrame([{
        "barrio": "TODOS",
        "alojamientos": listings["listing_id"].nunique(),
        "capacidad_total": listings["accommodates"].sum(),
        "capacidad_media": listings["accommodates"].mean(),
        "capacidad_mediana": listings["accommodates"].median(),
        "capacidad_valida": listings["accommodates"].count()
    }])

    context = pd.concat(
        [regional, city],
        ignore_index=True
    )

    context.to_parquet(
        DATA / "market_context.parquet",
        index=False
    )

    calendar = pd.read_csv(
        RAW / "calendar.csv.gz",
        usecols=[
            "listing_id",
            "date",
            "available"
        ]
    )

    calendar["fecha"] = pd.to_datetime(
        calendar["date"],
        errors="coerce"
    ).dt.normalize()

    calendar = calendar.dropna(
        subset=["fecha", "listing_id"]
    )

    calendar["disponible"] = (
        calendar["available"]
        .astype(str)
        .str.lower()
        .map({
            "t": 1,
            "true": 1,
            "f": 0,
            "false": 0
        })
    )

    calendar = calendar.merge(
        listings[[
            "listing_id",
            "barrio",
            "accommodates"
        ]],
        on="listing_id",
        how="inner",
        validate="many_to_one"
    )

    if calendar.duplicated(
        ["listing_id", "fecha"]
    ).any():
        raise ValueError(
            "Duplicados en calendario"
        )

    calendar["plazas_disponibles"] = (
        calendar["accommodates"]
        * calendar["disponible"]
    )

    def aggregate(frame, keys):

        result = frame.groupby(keys).agg(
            anuncios=("listing_id", "nunique"),
            registros_validos=("disponible", "count"),
            disponibles=("disponible", "sum"),
            plazas_disponibles=("plazas_disponibles", "sum")
        ).reset_index()

        result["no_disponibles"] = (
            result["registros_validos"]
            - result["disponibles"]
        )

        result["disponibilidad_pct"] = np.where(
            result["registros_validos"] > 0,
            100 * result["disponibles"]
            / result["registros_validos"],
            np.nan
        )

        return result

    regional_daily = aggregate(
        calendar,
        ["barrio", "fecha"]
    )

    city_daily = aggregate(
        calendar,
        ["fecha"]
    )

    city_daily["barrio"] = "TODOS"

    daily = pd.concat(
        [regional_daily, city_daily],
        ignore_index=True
    )

    daily.to_parquet(
        DATA / "calendar_daily_context.parquet",
        index=False
    )

    return context, daily


def period_context(
    daily,
    barrio,
    target,
    frequency
):

    target = pd.Timestamp(target)

    if frequency == "diaria":

        end = target

    elif frequency == "semanal":

        end = target + pd.Timedelta(days=6)

    elif frequency == "mensual":

        end = target + pd.offsets.MonthEnd(0)

    else:

        raise ValueError("Frecuencia invalida")

    expected = len(
        pd.date_range(target, end, freq="D")
    )

    d = daily[
        (daily["barrio"] == barrio)
        & daily["fecha"].between(target, end)
    ]

    observed = d["fecha"].nunique()

    valid = d["registros_validos"].sum()

    available = d["disponibles"].sum()

    complete = (
        observed == expected
        and valid > 0
    )

    return {
        "dias_calendario": observed,
        "dias_esperados": expected,
        "calendario_completo": complete,
        "disponibilidad_media_pct": (
            100 * available / valid
            if complete else np.nan
        ),
        "noches_anuncio_disponibles": (
            available if complete else np.nan
        ),
        "noches_anuncio_no_disponibles": (
            valid - available
            if complete else np.nan
        )
    }


def main():

    print("=" * 72)
    print("VALENCIA AIRBNB INTELLIGENCE LAB 3.0")
    print("ESTIMACION DE ESTANCIAS Y HUESPEDES")
    print("=" * 72)

    market, daily = prepare_market_context()

    print("\nCONTEXTO DE MERCADO")

    print("Ambitos:", len(market))

    print(
        "Registros diarios:",
        len(daily)
    )

    final = pd.read_csv(
        OUT / "forecast_final_predictions.csv"
    )

    results = []

    for _, row in final.iterrows():

        if not np.isfinite(
            row["inferior_95"]
        ):
            continue

        if not np.isfinite(
            row["superior_95"]
        ):
            continue

        scenarios = estimate_scenarios(
            row["prediccion"],
            row["inferior_95"],
            row["superior_95"]
        )

        context = period_context(
            daily,
            row["barrio"],
            row["objetivo"],
            row["frecuencia"]
        )

        results.append({
            **row.to_dict(),
            **scenarios,
            **context
        })

    report = pd.DataFrame(results)

    report = report.merge(
        market,
        on="barrio",
        how="left",
        validate="many_to_one"
    )

    report.to_csv(
        OUT / "forecast_tourism_estimates.csv",
        index=False
    )

    columns = [
        "barrio",
        "frecuencia",
        "prediccion",
        "estancias_central",
        "huespedes_central",
        "huespedes_escenario_bajo",
        "huespedes_escenario_alto",
        "disponibilidad_media_pct"
    ]

    selected = report[
        report["barrio"].isin([
            "BENIFERRI",
            "RUSSAFA",
            "CABANYAL-CANYAMELAR",
            "TODOS"
        ])
    ]

    print("\nRESULTADOS PRINCIPALES")

    print(
        selected[columns]
        .round(2)
        .to_string(index=False)
    )

    print("\nRESUMEN")

    print(
        "Estimaciones generadas:",
        len(report)
    )

    print(
        "Calendarios completos:",
        report["calendario_completo"].sum()
    )

    print("\nSUPUESTOS CENTRALES")

    print("Tasa de resena: 50%")
    print("Huespedes por estancia: 2")

    print("\nADVERTENCIAS")

    print(
        "Las llegadas no representan personas unicas."
    )

    print(
        "No disponibilidad no equivale a reserva."
    )

    print(
        "Los escenarios no son intervalos del 95%."
    )

    print(
        "\nESTIMACION TURISTICA COMPLETADA"
    )


if __name__ == "__main__":
    main()
