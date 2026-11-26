from pathlib import Path
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw"
DEST.mkdir(parents=True, exist_ok=True)

PAGE = "https://insideairbnb.com/get-the-data/"
DATE = "2026-09-20"

FILES = [
    "listings.csv.gz",
    "calendar.csv.gz",
    "reviews.csv.gz",
    "neighbourhoods.csv",
    "neighbourhoods.geojson",
]

response = requests.get(PAGE, timeout=60)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

links = [
    urljoin(PAGE, a["href"])
    for a in soup.find_all("a", href=True)
]

for filename in FILES:

    candidates = [
        url for url in links
        if "/valencia/" in url.lower()
        and DATE in url
        and url.endswith("/" + filename)
    ]

    if not candidates:
        print(f"ENLACE NO ENCONTRADO: {filename}")
        continue

    url = candidates[0]
    target = DEST / filename
    temporary = DEST / (filename + ".part")

    if target.exists() and target.stat().st_size > 0:
        print(f"YA EXISTE: {filename}")
        continue

    print(f"Descargando {filename}...")

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()

        with temporary.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)

    temporary.replace(target)

    mb = target.stat().st_size / 1024**2

    print(f"OK: {filename} | {mb:.2f} MB")

print("\nPROCESO DE DESCARGA FINALIZADO")

for file in DEST.iterdir():
    print(file.name)
