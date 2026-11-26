from pathlib import Path
import gzip
import hashlib
import json
import time

import pandas as pd
import numpy as np

from src.analytics import (
    descriptive_statistics,
    temporal_reviews,
    moving_average,
    temporal_decomposition,
    autocorrelation_analysis,
    available_stay
)

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "tables"

OUT.mkdir(parents=True, exist_ok=True)

results = []
start = time.perf_counter()


def check(name, condition, detail=""):

    passed = bool(condition)

    results.append({
        "prueba": name,
        "estado": "PASS" if passed else "FAIL",
        "detalle": str(detail)
    })

    print(
        f"[{'PASS' if passed else 'FAIL'}] "
        f"{name} | {detail}"
    )


print("=" * 75)
print("VALENCIA AIRBNB INTELLIGENCE LAB")
print("AUDITORIA INTEGRAL - VERSION 2.0")
print("=" * 75)

# =====================================
# 1. ARCHIVOS DEL PROYECTO
# =====================================

print("\n1. ARCHIVOS DEL PROYECTO")

required = [
    "app/app.py",
    "app/advanced_tabs.py",
    "app/geographic.py",
    "src/analytics.py",
    "src/download_data.py",
    "src/inspect_data.py",
    "src/query_engine.py",
    "src/audit_reviews.py",
    "test_analytics.py",
    "requirements.txt",
    ".gitignore",
    ".streamlit/config.toml",
    "data/raw/listings.csv.gz",
    "data/raw/calendar.csv.gz",
    "data/raw/reviews.csv.gz",
    "data/raw/neighbourhoods.csv",
    "data/raw/neighbourhoods.geojson"
]

for filename in required:
    check(
        filename,
        (ROOT / filename).exists()
    )

# =====================================
# 2. INTEGRIDAD DE ARCHIVOS
# =====================================

print("\n2. INTEGRIDAD DE DATOS COMPRIMIDOS")

hashes = {}

for filename in [
    "listings.csv.gz",
    "calendar.csv.gz",
    "reviews.csv.gz"
]:

    path = RAW / filename

    try:

        with gzip.open(path, "rb") as file:

            while file.read(1024 * 1024):
                pass

        check(
            f"Integridad gzip: {filename}",
            True
        )

        sha256 = hashlib.sha256()

        with path.open("rb") as file:

            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b""
            ):
                sha256.update(chunk)

        hashes[filename] = sha256.hexdigest()

    except Exception as error:

        check(filename, False, error)

# =====================================
# 3. CARGA DE DATOS
# =====================================

print("\n3. CARGA DE DATOS REALES")

listings = pd.read_csv(
    RAW / "listings.csv.gz",
    low_memory=False
)

calendar = pd.read_csv(
    RAW / "calendar.csv.gz"
)

reviews = pd.read_csv(
    RAW / "reviews.csv.gz",
    usecols=[
        "listing_id",
        "id",
        "date",
        "reviewer_id"
    ]
)

calendar["date"] = pd.to_datetime(
    calendar["date"]
)

reviews["date"] = pd.to_datetime(
    reviews["date"]
)

calendar["available_numeric"] = (
    calendar["available"].map({
        "t": 1,
        "f": 0
    })
)

listings["price_numeric"] = pd.to_numeric(
    listings["price"]
    .astype("string")
    .str.replace(
        r"[^0-9.,-]", "", regex=True
    )
    .str.replace(
        ",", "", regex=False
    ),
    errors="coerce"
)

# =====================================
# 4. DIMENSIONES CONOCIDAS
# =====================================

print("\n4. DIMENSIONES Y REGRESION")

check(
    "Total alojamientos",
    len(listings) == 7968,
    len(listings)
)

check(
    "Total calendario",
    len(calendar) == 2908320,
    len(calendar)
)

check(
    "Total asociaciones de resenas",
    len(reviews) == 509513,
    len(reviews)
)

check(
    "Total barrios",
    listings["neighbourhood_cleansed"].nunique() == 84,
    listings["neighbourhood_cleansed"].nunique()
)

check(
    "Total anfitriones",
    listings["host_id"].nunique() == 3215,
    listings["host_id"].nunique()
)

check(
    "Precios validos",
    listings["price_numeric"].count() == 7197,
    listings["price_numeric"].count()
)

check(
    "Mediana de precios",
    np.isclose(
        listings["price_numeric"].median(),
        134.10
    ),
    listings["price_numeric"].median()
)

# =====================================
# 5. INTEGRIDAD RELACIONAL
# =====================================

print("\n5. INTEGRIDAD RELACIONAL")

listing_ids = set(listings["id"])
calendar_ids = set(calendar["listing_id"])
review_ids = set(reviews["listing_id"])

check(
    "ID de alojamiento unico",
    listings["id"].is_unique
)

check(
    "Correspondencia listings-calendar",
    listing_ids == calendar_ids
)

check(
    "Reviews asociados a listings",
    review_ids.issubset(listing_ids)
)

check(
    "Clave compuesta calendario",
    not calendar.duplicated(
        ["listing_id", "date"]
    ).any()
)

check(
    "Clave compuesta resenas",
    not reviews.duplicated(
        ["listing_id", "id"]
    ).any()
)

check(
    "Resenas con ID unico",
    reviews["id"].nunique() == 458154,
    reviews["id"].nunique()
)

declared = listings.set_index(
    "id"
)["number_of_reviews"]

observed = (
    reviews.groupby("listing_id")
    .size()
    .reindex(declared.index, fill_value=0)
)

check(
    "Conteo de resenas consistente",
    declared.eq(observed).all(),
    f"{declared.eq(observed).sum()} coincidencias"
)

# =====================================
# 6. VALIDACION TEMPORAL
# =====================================

print("\n6. VALIDACION TEMPORAL")

check(
    "365 fechas de calendario",
    calendar["date"].nunique() == 365
)

check(
    "365 registros por alojamiento",
    calendar.groupby(
        "listing_id"
    ).size().eq(365).all()
)

check(
    "Disponibilidad binaria valida",
    calendar["available_numeric"]
    .isin([0, 1]).all()
)

availability = (
    calendar["available_numeric"].mean()
    * 100
)

check(
    "Disponibilidad general",
    round(availability, 2) == 52.27,
    round(availability, 2)
)

daily = calendar.groupby(
    "date"
)["available_numeric"].sum()

check(
    "Maxima disponibilidad diaria",
    daily.max() == 5889,
    daily.max()
)

check(
    "Minima disponibilidad diaria",
    daily.min() == 528,
    daily.min()
)

# =====================================
# 7. VALIDACION GEOGRAFICA
# =====================================

print("\n7. VALIDACION GEOGRAFICA")

check(
    "Latitudes validas",
    listings["latitude"].between(
        -90, 90
    ).all()
)

check(
    "Longitudes validas",
    listings["longitude"].between(
        -180, 180
    ).all()
)

geojson = json.loads(
    (
        RAW / "neighbourhoods.geojson"
    ).read_text(encoding="utf-8-sig")
)

check(
    "GeoJSON valido",
    geojson.get("type") == "FeatureCollection"
    and len(geojson.get("features", [])) > 0,
    f"{len(geojson.get('features', []))} elementos"
)

# =====================================
# 8. CONSULTAS POR BARRIO
# =====================================

print("\n8. CONSULTAS POR BARRIO")

expected = {
    "BENIFERRI": 10,
    "BENICALAP": 128,
    "CAMPANAR": 46,
    "RUSSAFA": 551,
    "ALBORS": 102,
    "CABANYAL-CANYAMELAR": 774
}

for barrio, count in expected.items():

    actual = listings[
        "neighbourhood_cleansed"
    ].eq(barrio).sum()

    check(
        f"Barrio {barrio}",
        actual == count,
        actual
    )

# =====================================
# 9. MATEMATICAS CON DATOS REALES
# =====================================

print("\n9. FUNCIONES MATEMATICAS")

stats = descriptive_statistics(
    listings["price_numeric"]
)

check(
    "Estadistica descriptiva",
    np.isclose(
        stats["Mediana"],
        134.10
    )
)

series = temporal_reviews(
    reviews,
    frequency="MS",
    unique=True
)

# Excluir meses extremos

series = series.iloc[1:-1]

check(
    "Serie temporal mensual",
    len(series) >= 24,
    len(series)
)

moving = moving_average(series, 3)

check(
    "Media movil",
    np.isclose(
        moving.iloc[-1],
        series.iloc[-3:].mean()
    )
)

decomposition = temporal_decomposition(
    series,
    period=12
)

if decomposition is not None:

    reconstructed = (
        decomposition["tendencia"]
        + decomposition["estacionalidad"]
        + decomposition["residuo"]
    )

    error = np.max(np.abs(
        decomposition["observado"]
        - reconstructed
    ))

    check(
        "Reconstruccion STL",
        error < 1e-8,
        f"Error: {error:.3e}"
    )

else:

    check(
        "Descomposicion STL",
        False,
        "Datos insuficientes"
    )

correlations = autocorrelation_analysis(
    series
)

check(
    "ACF retardo cero",
    correlations is not None
    and np.isclose(
        correlations["ACF"].iloc[0], 1
    )
)

check(
    "PACF retardo cero",
    correlations is not None
    and np.isclose(
        correlations["PACF"].iloc[0], 1
    )
)

# =====================================
# 10. DISPONIBILIDAD CONSECUTIVA
# =====================================

print("\n10. ESTANCIAS COMPLETAS")

date = pd.Timestamp("2026-12-14")
nights = 3

compatible = set(
    available_stay(
        calendar, date, nights
    )
)

interval = calendar[
    (calendar["date"] >= date)
    & (
        calendar["date"]
        < date + pd.Timedelta(days=nights)
    )
]

matrix = interval.pivot(
    index="listing_id",
    columns="date",
    values="available_numeric"
)

expected_available = set(
    matrix.index[matrix.eq(1).all(axis=1)]
)

restrictions = calendar[
    (calendar["date"] == date)
    & (calendar["minimum_nights"] <= nights)
    & (calendar["maximum_nights"] >= nights)
]

expected_available &= set(
    restrictions["listing_id"]
)

check(
    "Validacion independiente de estancia",
    compatible == expected_available,
    f"{len(compatible)} alojamientos compatibles"
)

# =====================================
# 11. INFORME FINAL
# =====================================

report = pd.DataFrame(results)

report.to_csv(
    OUT / "system_validation.csv",
    index=False,
    encoding="utf-8-sig"
)

(OUT / "data_hashes.json").write_text(
    json.dumps(hashes, indent=4),
    encoding="utf-8"
)

passed = report["estado"].eq("PASS").sum()
failed = report["estado"].eq("FAIL").sum()

elapsed = time.perf_counter() - start

print("\n" + "=" * 75)
print("RESULTADO FINAL")
print("=" * 75)

print("Pruebas totales:", len(report))
print("Pruebas superadas:", passed)
print("Pruebas fallidas:", failed)
print(f"Tiempo de ejecucion: {elapsed:.2f} segundos")

print("\nInforme:", OUT / "system_validation.csv")

if failed:
    print("\nEXISTEN PRUEBAS PENDIENTES DE CORRECCION")
    raise SystemExit(1)

print("\nTODAS LAS VALIDACIONES SUPERADAS")
