from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "tables"

OUT.mkdir(parents=True, exist_ok=True)

datasets = {
    "listings": "listings.csv.gz",
    "calendar": "calendar.csv.gz",
    "reviews": "reviews.csv.gz",
}

summary = []

for name, filename in datasets.items():

    print("\n" + "=" * 65)
    print(f"DATASET: {name.upper()}")
    print("=" * 65)

    df = pd.read_csv(
        RAW / filename,
        low_memory=False
    )

    duplicates = int(df.duplicated().sum())

    print(f"Filas: {len(df):,}")
    print(f"Columnas: {len(df.columns)}")
    print(f"Duplicados: {duplicates:,}")

    audit = pd.DataFrame({
        "variable": df.columns,
        "tipo": df.dtypes.astype(str).values,
        "nulos": df.isna().sum().values,
        "porcentaje_nulos": (
            df.isna().mean().values * 100
        ).round(2),
        "valores_unicos": df.nunique().values
    })

    print("\nDICCIONARIO DE VARIABLES:")
    print(audit.to_string(index=False))

    audit.to_csv(
        OUT / f"audit_{name}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary.append({
        "dataset": name,
        "filas": len(df),
        "columnas": len(df.columns),
        "duplicados": duplicates
    })

    if name == "listings":

        if "neighbourhood_cleansed" in df.columns:

            barrios = df["neighbourhood_cleansed"].value_counts()

            print("\nTOTAL DE BARRIOS:", len(barrios))
            print("\nALOJAMIENTOS POR BARRIO:")
            print(barrios.to_string())

            barrios.to_csv(
                OUT / "neighbourhood_counts.csv",
                header=["alojamientos"],
                encoding="utf-8-sig"
            )

        if "room_type" in df.columns:
            print("\nTIPOS DE ALOJAMIENTO:")
            print(df["room_type"].value_counts())

        if "id" in df.columns:
            print("\nIDs UNICOS:", df["id"].nunique())
            print("IDs DUPLICADOS:", df["id"].duplicated().sum())

    if "date" in df.columns:

        dates = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        print("\nRANGO TEMPORAL:")
        print("Fecha inicial:", dates.min())
        print("Fecha final:", dates.max())
        print("Fechas invalidas:", dates.isna().sum())

    if "listing_id" in df.columns:
        print("\nALOJAMIENTOS UNICOS:", df["listing_id"].nunique())

    if name == "calendar":

        if "available" in df.columns:
            print("\nDISPONIBILIDAD:")
            print(df["available"].value_counts(dropna=False))

        if {"listing_id", "date"}.issubset(df.columns):
            print("\nDuplicados alojamiento-fecha:")
            print(df.duplicated(["listing_id", "date"]).sum())

    del df

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    OUT / "dataset_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 65)
print("RESUMEN GENERAL")
print("=" * 65)
print(summary_df.to_string(index=False))

print("\nAUDITORIA FINALIZADA CORRECTAMENTE")
print("Resultados guardados en:", OUT)
