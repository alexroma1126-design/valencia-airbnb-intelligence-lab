from pathlib import Path
import json

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "informes" / "figuras"

OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "figure.facecolor": "white",
    "savefig.dpi": 300
})

BLUE = "#173B63"
CYAN = "#3288BD"
GREEN = "#23856D"
ORANGE = "#D97706"


def save(fig, filename):
    fig.tight_layout()
    fig.savefig(
        OUT / filename,
        bbox_inches="tight",
        dpi=300
    )
    plt.close(fig)
    print("[OK]", filename)


print("CARGANDO BASES DE DATOS")

l = pd.read_csv(
    RAW / "listings.csv.gz",
    low_memory=False
)

c = pd.read_csv(
    RAW / "calendar.csv.gz"
)

r = pd.read_csv(
    RAW / "reviews.csv.gz",
    usecols=["listing_id", "id", "date"]
)

l["price_numeric"] = pd.to_numeric(
    l["price"].astype("string")
    .str.replace(r"[^0-9.,-]", "", regex=True)
    .str.replace(",", "", regex=False),
    errors="coerce"
)

c["date"] = pd.to_datetime(c["date"])
r["date"] = pd.to_datetime(r["date"])

c["available_numeric"] = c["available"].map({
    "t": 1,
    "f": 0
})

# 01. CALIDAD DE DATOS

missing = (
    l.drop(columns=["price_numeric"])
    .isna().mean().mul(100)
    .sort_values(ascending=False)
    .head(15)
    .sort_values()
)

fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(missing.index, missing.values, color=BLUE)

ax.set_title("Variables con mayor porcentaje de datos faltantes")
ax.set_xlabel("Valores faltantes (%)")

save(fig, "01_missing_values.png")

# 02. DISTRIBUCION POR BARRIOS

counts = (
    l["neighbourhood_cleansed"]
    .value_counts().head(20)
    .sort_values()
)

fig, ax = plt.subplots(figsize=(10, 7))

ax.barh(counts.index, counts.values, color=BLUE)

ax.set_title("Veinte barrios con mayor numero de anuncios")
ax.set_xlabel("Alojamientos")

save(fig, "02_neighbourhoods.png")

# 03. TIPOS DE ALOJAMIENTO

rooms = l["room_type"].value_counts()

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(
    rooms.index, rooms.values,
    color=[BLUE, CYAN, GREEN, ORANGE][:len(rooms)]
)

ax.bar_label(bars, padding=4)
ax.set_ylabel("Numero de anuncios")
ax.set_title("Composicion de la oferta de alojamientos")

save(fig, "03_room_types.png")

# 04. DISTRIBUCION DE PRECIOS

prices = l["price_numeric"].dropna()
q99 = prices.quantile(0.99)

visible = prices[prices <= q99]

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(
    visible, bins=60,
    color=BLUE,
    edgecolor="white"
)

ax.axvline(
    prices.median(),
    color=ORANGE,
    linewidth=2,
    linestyle="--",
    label=f"Mediana global: {prices.median():.2f}"
)

ax.axvline(
    prices.mean(),
    color=GREEN,
    linewidth=2,
    linestyle="--",
    label=f"Media global: {prices.mean():.2f}"
)

ax.set_title("Distribucion de precios registrados hasta P99")
ax.set_xlabel("Precio registrado")
ax.set_ylabel("Frecuencia")
ax.legend()

save(fig, "04_price_distribution.png")

# 05. COMPARACION DE PRECIOS POR BARRIO

top = (
    l["neighbourhood_cleansed"]
    .value_counts().head(10).index
)

groups = [
    l.loc[
        l["neighbourhood_cleansed"].eq(b),
        "price_numeric"
    ].dropna().values
    for b in top
]

fig, ax = plt.subplots(figsize=(11, 6))

bp = ax.boxplot(
    groups,
    showfliers=False,
    patch_artist=True
)

for box in bp["boxes"]:
    box.set_facecolor(CYAN)
    box.set_alpha(0.65)

ax.set_xticks(range(1, len(top) + 1))
ax.set_xticklabels(top, rotation=40, ha="right")

ax.set_title("Comparacion de precios en los diez barrios principales")
ax.set_ylabel("Precio registrado")

save(fig, "05_price_boxplots.png")

# 06. MAPA GEOGRAFICO LOCAL

geo = json.loads(
    (RAW / "neighbourhoods.geojson")
    .read_text(encoding="utf-8-sig")
)

fig, ax = plt.subplots(figsize=(9, 9))

for feature in geo["features"]:

    geometry = feature.get("geometry")

    if not geometry:
        continue

    coords = geometry["coordinates"]

    if geometry["type"] == "Polygon":
        polygons = [coords]
    elif geometry["type"] == "MultiPolygon":
        polygons = coords
    else:
        continue

    for polygon in polygons:
        for ring in polygon:

            xy = np.asarray(ring)

            ax.plot(
                xy[:, 0],
                xy[:, 1],
                color="#A5B4C5",
                linewidth=0.65
            )

colors = {
    "Entire home/apt": BLUE,
    "Private room": GREEN,
    "Shared room": ORANGE,
    "Hotel room": "#A43D54"
}

for room, group in l.groupby("room_type"):

    ax.scatter(
        group["longitude"],
        group["latitude"],
        s=5,
        alpha=0.45,
        color=colors.get(room, BLUE),
        label=room
    )

lat = l["latitude"].median()

ax.set_aspect(1 / np.cos(np.deg2rad(lat)))

ax.set_title("Distribucion geografica de alojamientos en Valencia")
ax.set_xlabel("Longitud")

ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
ax.tick_params(axis="both", labelsize=9)

ax.tick_params(axis="x", labelrotation=25)
ax.set_ylabel("Latitud")

ax.legend(markerscale=3, loc="best")

save(fig, "06_geographic_map.png")

# 07. DISPONIBILIDAD DIARIA

daily = (
    c.groupby("date")["available_numeric"]
    .mean().mul(100)
)

fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(
    daily.index, daily.values,
    color=BLUE,
    linewidth=1.3,
    label="Disponibilidad diaria"
)

ax.plot(
    daily.index,
    daily.rolling(7).mean(),
    color=ORANGE,
    linewidth=2,
    label="Media movil de siete dias"
)

ax.axhline(
    daily.mean(),
    linestyle="--",
    color=GREEN,
    label=f"Promedio: {daily.mean():.2f}%"
)

ax.set_title("Disponibilidad anunciada en el calendario")
ax.set_ylabel("Alojamientos disponibles (%)")
ax.legend()

fig.autofmt_xdate()

save(fig, "07_daily_availability.png")

# 08. RESEÑAS HISTORICAS

monthly = (
    r.groupby(pd.Grouper(key="date", freq="MS"))
    .agg(
        unique_reviews=("id", "nunique"),
        associations=("id", "size")
    )
    .iloc[1:-1]
)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    monthly.index,
    monthly["associations"],
    color=CYAN,
    alpha=0.7,
    label="Asociaciones alojamiento-resena"
)

ax.plot(
    monthly.index,
    monthly["unique_reviews"],
    color=BLUE,
    label="Identificadores unicos"
)

ax.plot(
    monthly.index,
    monthly["unique_reviews"].rolling(12).mean(),
    color=ORANGE,
    linewidth=2,
    label="Media movil de doce meses"
)

ax.set_title("Evolucion mensual de las resenas registradas")
ax.set_ylabel("Numero de registros")
ax.legend()

save(fig, "08_monthly_reviews.png")

# 09. DESCOMPOSICION STL

series = monthly["unique_reviews"].astype(float)

stl = STL(
    series,
    period=12,
    robust=True
).fit()

fig, axes = plt.subplots(
    4, 1,
    figsize=(12, 9),
    sharex=True
)

components = [
    ("Serie observada", series),
    ("Tendencia", stl.trend),
    ("Estacionalidad", stl.seasonal),
    ("Residuo", stl.resid)
]

for ax, (name, values) in zip(axes, components):

    ax.plot(values.index, values.values, color=BLUE)
    ax.set_ylabel(name)

axes[0].set_title("Descomposicion STL de las resenas mensuales")

save(fig, "09_stl_decomposition.png")

# 10. AUTOCORRELACION

fig, axes = plt.subplots(
    2, 1,
    figsize=(11, 7)
)

plot_acf(
    series,
    lags=36,
    ax=axes[0],
    color=BLUE
)

plot_pacf(
    series,
    lags=36,
    ax=axes[1],
    method="ywm",
    color=BLUE
)

axes[0].set_title("Funcion de autocorrelacion (ACF)")
axes[1].set_title("Funcion de autocorrelacion parcial (PACF)")

axes[1].set_xlabel("Retardo mensual")

save(fig, "10_acf_pacf.png")

print("\nFIGURAS CIENTIFICAS GENERADAS")
print("Directorio:", OUT)
print("Total:", len(list(OUT.glob("*.png"))))
