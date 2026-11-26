from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "tables"

reviews = pd.read_csv(
    RAW / "reviews.csv.gz",
    low_memory=False
)

print("=" * 60)
print("AUDITORIA AVANZADA DE REVIEWS")
print("=" * 60)

print("\nRegistros totales:", len(reviews))

print("IDs unicos:", reviews["id"].nunique())

print(
    "Pares listing_id e id unicos:",
    len(reviews.drop_duplicates(["listing_id", "id"]))
)

print(
    "Pares listing_id e id repetidos:",
    reviews.duplicated(["listing_id", "id"]).sum()
)

repeated = reviews[
    reviews["id"].duplicated(keep=False)
].copy()

audit = repeated.groupby("id").agg(
    registros=("id", "size"),
    alojamientos=("listing_id", "nunique"),
    fechas=("date", "nunique"),
    huespedes=("reviewer_id", "nunique"),
    comentarios=("comments", "nunique")
)

print("\nIDENTIFICADORES REPETIDOS:")

print("Total de IDs afectados:", len(audit))

for col in [
    "alojamientos",
    "fechas",
    "huespedes",
    "comentarios"
]:
    print(
        f"IDs con multiples {col}:",
        (audit[col] > 1).sum()
    )

print("\nPRIMEROS 15 IDENTIFICADORES REPETIDOS:")
print(audit.head(15).to_string())

print("\nCOMPARACION CON LISTINGS")

listings = pd.read_csv(
    RAW / "listings.csv.gz",
    usecols=["id", "number_of_reviews"]
)

counts = reviews.groupby("listing_id").size()

unique_counts = (
    reviews.groupby("listing_id")["id"].nunique()
)

listings["registros_reviews"] = (
    listings["id"].map(counts).fillna(0)
)

listings["ids_reviews_unicos"] = (
    listings["id"].map(unique_counts).fillna(0)
)

listings["diferencia_registros"] = (
    listings["registros_reviews"]
    - listings["number_of_reviews"]
)

listings["diferencia_ids_unicos"] = (
    listings["ids_reviews_unicos"]
    - listings["number_of_reviews"]
)

print(
    "Total declarado en listings:",
    listings["number_of_reviews"].sum()
)

print(
    "Total de registros reviews:",
    listings["registros_reviews"].sum()
)

print(
    "Total de IDs unicos por alojamiento:",
    listings["ids_reviews_unicos"].sum()
)

print(
    "Coincidencias usando registros:",
    (listings["diferencia_registros"] == 0).sum()
)

print(
    "Coincidencias usando IDs unicos:",
    (listings["diferencia_ids_unicos"] == 0).sum()
)

audit.to_csv(OUT / "review_id_audit.csv")

listings.to_csv(
    OUT / "review_count_validation.csv",
    index=False
)

print("\nAUDITORIA FINALIZADA")
