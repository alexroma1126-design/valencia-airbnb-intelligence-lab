from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "app_data"
REPORTS = ROOT / "results" / "tables"

OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def load_data():

    listings = pd.read_csv(
        RAW / "listings.csv.gz",
        usecols=["id", "neighbourhood_cleansed"]
    )

    reviews = pd.read_csv(
        RAW / "reviews.csv.gz",
        usecols=["listing_id", "id", "date"]
    )

    reviews["date"] = pd.to_datetime(
        reviews["date"],
        errors="coerce"
    ).dt.normalize()

    reviews = reviews.dropna(
        subset=["id", "listing_id", "date"]
    )

    listings = listings.rename(columns={
        "id": "listing_id",
        "neighbourhood_cleansed": "barrio"
    })

    if listings["listing_id"].duplicated().any():
        raise ValueError("Listings contiene IDs duplicados")

    merged = reviews.merge(
        listings,
        on="listing_id",
        how="left",
        validate="many_to_one"
    )

    unmatched = int(merged["barrio"].isna().sum())

    merged = merged.dropna(subset=["barrio"])

    return listings, reviews, merged, unmatched


def build_series(reviews, frequency, cutoff, scope):

    records = []

    if scope == "TODOS":
        groups = [("TODOS", reviews)]
    else:
        groups = reviews.groupby("barrio")

    for barrio, group in groups:

        group = group.sort_values(
            ["date", "listing_id"]
        )

        group = group.drop_duplicates(
            subset=["id"]
        )

        dates = group["date"]

        if dates.empty:
            continue

        first = dates.min()

        if frequency == "D":

            period = dates.dt.normalize()
            start = first
            end = cutoff
            index_freq = "D"

        elif frequency == "W":

            period = dates.dt.to_period(
                "W-SUN"
            ).dt.start_time

            start = first.to_period(
                "W-SUN"
            ).start_time

            if first != start:
                start += pd.Timedelta(days=7)

            end = cutoff.to_period(
                "W-SUN"
            ).start_time

            if end + pd.Timedelta(days=6) > cutoff:
                end -= pd.Timedelta(days=7)

            index_freq = "W-MON"

        elif frequency == "M":

            period = dates.dt.to_period(
                "M"
            ).dt.start_time

            start = first.to_period("M").start_time

            if first != start:
                start += pd.offsets.MonthBegin(1)

            end = cutoff.to_period("M").start_time

            month_end = cutoff.to_period(
                "M"
            ).end_time.normalize()

            if cutoff < month_end:
                end -= pd.offsets.MonthBegin(1)

            index_freq = "MS"

        else:
            raise ValueError("Frecuencia no reconocida")

        if start > end:
            continue

        counts = period.value_counts().sort_index()

        index = pd.date_range(
            start=start,
            end=end,
            freq=index_freq
        )

        counts = counts.reindex(
            index,
            fill_value=0
        )

        frame = pd.DataFrame({
            "fecha": index,
            "barrio": barrio,
            "resenas": counts.to_numpy(dtype="int64")
        })

        records.append(frame)

    if not records:
        return pd.DataFrame(
            columns=["fecha", "barrio", "resenas"]
        )

    return pd.concat(
        records,
        ignore_index=True
    )


def main():

    print("=" * 65)
    print("VALENCIA AIRBNB INTELLIGENCE LAB 3.0")
    print("PREPARACION DE SERIES TEMPORALES")
    print("=" * 65)

    listings, reviews, merged, unmatched = load_data()

    last_date = reviews["date"].max()

    # Margen conservador configurable.
    # No representa una correccion demostrada del retraso.
    buffer_days = 14

    cutoff = last_date - pd.Timedelta(
        days=buffer_days
    )

    metadata = {
        "alojamientos": int(len(listings)),
        "barrios": int(listings["barrio"].nunique()),
        "filas_reviews_validas": int(len(reviews)),
        "resenas_unicas": int(reviews["id"].nunique()),
        "reviews_sin_barrio": unmatched,
        "fecha_inicial": str(reviews["date"].min().date()),
        "fecha_final_observada": str(last_date.date()),
        "margen_conservador_dias": buffer_days,
        "fecha_corte_modelado": str(cutoff.date()),
        "unidad_objetivo": "resenas unicas",
        "semana": "lunes a domingo",
        "fuente": "Inside Airbnb"
    }

    print("\nAUDITORIA DE DATOS")

    for key, value in metadata.items():
        print(f"{key}: {value}")

    summary = []

    frequencies = {
        "D": "diarias",
        "W": "semanales",
        "M": "mensuales"
    }

    for frequency, name in frequencies.items():

        regional = build_series(
            merged,
            frequency,
            cutoff,
            "BARRIOS"
        )

        city = build_series(
            merged,
            frequency,
            cutoff,
            "TODOS"
        )

        data = pd.concat(
            [regional, city],
            ignore_index=True
        ).sort_values(
            ["barrio", "fecha"]
        )

        assert not data.duplicated(
            ["barrio", "fecha"]
        ).any()

        assert data["resenas"].ge(0).all()

        assert data["fecha"].le(cutoff).all()

        file = OUT / f"reviews_{name}.parquet"

        data.to_parquet(
            file,
            index=False,
            compression="snappy"
        )

        print(f"\nARCHIVO: {file.name}")
        print(f"Filas: {len(data):,}")
        print(f"Ambitos: {data['barrio'].nunique()}")
        print(f"Inicio: {data['fecha'].min()}")
        print(f"Fin: {data['fecha'].max()}")
        print(f"Tamano: {file.stat().st_size / 1024:.2f} KB")

        summary.append({
            "frecuencia": name,
            "filas": len(data),
            "ambitos": data["barrio"].nunique(),
            "fecha_inicial": data["fecha"].min(),
            "fecha_final": data["fecha"].max(),
            "periodos_cero": int(
                data["resenas"].eq(0).sum()
            )
        })

    (OUT / "metadata.json").write_text(
        json.dumps(
            metadata,
            indent=4,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    pd.DataFrame(summary).to_csv(
        REPORTS / "forecast_data_summary.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("SERIES TEMPORALES GENERADAS CORRECTAMENTE")
    print("=" * 65)


if __name__ == "__main__":
    main()
