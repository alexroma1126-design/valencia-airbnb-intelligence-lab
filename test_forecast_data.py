from pathlib import Path
import pandas as pd
import json

root = Path.cwd()
folder = root / "app_data"

print("=" * 65)
print("VALIDACION DE SERIES TEMPORALES")
print("=" * 65)

meta = json.loads(
    (folder / "metadata.json").read_text(
        encoding="utf-8"
    )
)

print("\nMETADATOS:")
print(json.dumps(meta, indent=4))

files = {
    "reviews_diarias.parquet": "D",
    "reviews_semanales.parquet": "W-MON",
    "reviews_mensuales.parquet": "MS"
}

for filename, freq in files.items():

    df = pd.read_parquet(folder / filename)

    print("\n" + "-" * 65)
    print(filename)

    assert not df.duplicated(
        ["barrio", "fecha"]
    ).any()

    assert df["resenas"].ge(0).all()

    for barrio, group in df.groupby("barrio"):

        dates = pd.DatetimeIndex(
            group["fecha"].sort_values()
        )

        expected = pd.date_range(
            dates.min(),
            dates.max(),
            freq=freq
        )

        assert dates.equals(expected), barrio

    print("Continuidad temporal: OK")
    print("Duplicados: OK")
    print("Conteos no negativos: OK")
    print("Ambitos:", df["barrio"].nunique())

    for barrio in [
        "RUSSAFA",
        "CABANYAL-CANYAMELAR",
        "BENIFERRI",
        "TODOS"
    ]:

        selected = df[
            df["barrio"] == barrio
        ]

        if selected.empty:
            continue

        print(f"\nBARRIO: {barrio}")
        print("Observaciones:", len(selected))
        print(
            selected.tail(5).to_string(
                index=False
            )
        )

print("\nTODAS LAS VALIDACIONES SUPERADAS")
