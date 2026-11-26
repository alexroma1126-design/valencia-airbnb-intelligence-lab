from pathlib import Path
import pandas as pd
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "tables"

OUT.mkdir(parents=True, exist_ok=True)

print("Cargando bases de Valencia...")

listings = pd.read_csv(
    RAW / "listings.csv.gz",
    low_memory=False
)

calendar = pd.read_csv(
    RAW / "calendar.csv.gz",
    low_memory=False
)

reviews = pd.read_csv(
    RAW / "reviews.csv.gz",
    usecols=["listing_id", "id", "date"]
)

def normalize(value):
    value = str(value).strip().upper()
    return "".join(
        c for c in unicodedata.normalize("NFD", value)
        if unicodedata.category(c) != "Mn"
    )

# Preparacion de variables

listings["price_numeric"] = pd.to_numeric(
    listings["price"]
    .astype("string")
    .str.replace(r"[^0-9.,-]", "", regex=True)
    .str.replace(",", "", regex=False),
    errors="coerce"
)

calendar["date"] = pd.to_datetime(
    calendar["date"],
    errors="coerce"
)

calendar["available_numeric"] = (
    calendar["available"].map({"t": 1, "f": 0})
)

reviews["date"] = pd.to_datetime(
    reviews["date"],
    errors="coerce"
)

# Auditoria relacional

listing_ids = set(listings["id"])
calendar_ids = set(calendar["listing_id"])
review_ids = set(reviews["listing_id"])

print("\nAUDITORIA RELACIONAL")
print("IDs listings:", len(listing_ids))
print("IDs calendar:", len(calendar_ids))
print("IDs reviews:", len(review_ids))

print(
    "IDs calendar sin listing:",
    len(calendar_ids - listing_ids)
)

print(
    "IDs listings sin calendar:",
    len(listing_ids - calendar_ids)
)

print(
    "IDs reviews sin listing:",
    len(review_ids - listing_ids)
)

print(
    "Listings sin reviews:",
    len(listing_ids - review_ids)
)

print(
    "IDs de reviews repetidos:",
    reviews["id"].duplicated().sum()
)

print(
    "IDs de reviews asociados a varios alojamientos:",
    (reviews.groupby("id")["listing_id"].nunique() > 1).sum()
)

print(
    "Precios numericos validos:",
    listings["price_numeric"].notna().sum()
)

print("\nMUESTRA DEL FORMATO DE PRECIOS")
print(
    listings[["price", "price_numeric"]]
    .dropna()
    .head(10)
    .to_string(index=False)
)

# Funcion principal

def analyze_neighborhood(name):

    if normalize(name) == "TODOS":
        selected = listings.copy()
    else:
        selected = listings[
            listings["neighbourhood_cleansed"]
            .map(normalize) == normalize(name)
        ].copy()

    if selected.empty:
        print("\nBarrio no encontrado.")
        return

    ids = selected["id"]

    cal = calendar[
        calendar["listing_id"].isin(ids)
    ].copy()

    rev = reviews[
        reviews["listing_id"].isin(ids)
    ].copy()

    print("\n" + "=" * 60)
    print("ANALISIS:", name.upper())
    print("=" * 60)

    print("Alojamientos:", len(selected))
    print("Anfitriones:", selected["host_id"].nunique())
    print("Registros calendario:", len(cal))
    print("Registros reviews:", len(rev))

    print("\nTIPOS DE ALOJAMIENTO")
    print(selected["room_type"].value_counts())

    print("\nESTADISTICAS DE PRECIOS")
    print(
        selected["price_numeric"]
        .describe()
        .round(2)
        .to_string()
    )

    print("\nTOP 5 ALOJAMIENTOS MAS CAROS")

    columns = [
        "id",
        "name",
        "price_numeric",
        "listing_url"
    ]

    print(
        selected.dropna(subset=["price_numeric"])
        .nlargest(5, "price_numeric")[columns]
        .to_string(index=False)
    )

    # Serie temporal de disponibilidad

    daily = (
        cal.groupby("date")
        .agg(
            alojamientos=("listing_id", "nunique"),
            disponibles=("available_numeric", "sum"),
            registros_validos=("available_numeric", "count")
        )
        .reset_index()
    )

    daily["no_disponibles"] = (
        daily["registros_validos"] - daily["disponibles"]
    )

    daily["disponibilidad_pct"] = (
        daily["disponibles"]
        / daily["registros_validos"] * 100
    )

    print("\nDISPONIBILIDAD MEDIA (%)")
    print(round(daily["disponibilidad_pct"].mean(), 2))

    print("\nFECHAS CON MAYOR DISPONIBILIDAD")
    print(
        daily.nlargest(5, "disponibilidad_pct")
        .to_string(index=False)
    )

    print("\nFECHAS CON MENOR DISPONIBILIDAD")
    print(
        daily.nsmallest(5, "disponibilidad_pct")
        .to_string(index=False)
    )

    # Serie temporal de resenas

    monthly = (
        rev.groupby(pd.Grouper(key="date", freq="MS"))
        .agg(
            registros=("id", "size"),
            ids_unicos=("id", "nunique")
        )
        .reset_index()
    )

    print("\nULTIMOS MESES DE RESENAS")
    print(monthly.tail(12).to_string(index=False))

    prefix = "".join(
        c if c.isalnum() else "_"
        for c in normalize(name)
    ).lower()

    daily.to_csv(
        OUT / f"{prefix}_daily_availability.csv",
        index=False
    )

    monthly.to_csv(
        OUT / f"{prefix}_monthly_reviews.csv",
        index=False
    )

    selected.drop(columns=["price_quote_raw"], errors="ignore")[
        [
            "id",
            "name",
            "neighbourhood_cleansed",
            "room_type",
            "price_numeric",
            "latitude",
            "longitude",
            "listing_url"
        ]
    ].to_csv(
        OUT / f"{prefix}_listings.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nResultados guardados correctamente.")

# Menu interactivo

while True:

    print("\n" + "=" * 60)
    print("VALENCIA AIRBNB INTELLIGENCE LAB")
    print("=" * 60)

    print("1. Mostrar barrios")
    print("2. Analizar un barrio")
    print("3. Analizar toda Valencia")
    print("0. Salir")

    option = input("\nSelecciona una opcion: ").strip()

    if option == "1":

        print(
            listings["neighbourhood_cleansed"]
            .value_counts()
            .to_string()
        )

    elif option == "2":

        barrio = input("Nombre del barrio: ")
        analyze_neighborhood(barrio)

    elif option == "3":

        analyze_neighborhood("TODOS")

    elif option == "0":

        print("Programa finalizado.")
        break

    else:

        print("Opcion no valida.")
