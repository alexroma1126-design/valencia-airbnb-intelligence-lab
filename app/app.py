from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

st.set_page_config(
    page_title="Valencia Airbnb Intelligence Lab",
    page_icon="🏙️",
    layout="wide"
)

@st.cache_data(show_spinner="Cargando datos de Valencia...")
def load_data():

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
        usecols=[
            "listing_id",
            "id",
            "date",
            "reviewer_id"
        ]
    )

    listings["price_numeric"] = pd.to_numeric(
        listings["price"]
        .astype("string")
        .str.replace(r"[^0-9.,-]", "", regex=True)
        .str.replace(",", "", regex=False),
        errors="coerce"
    )

    calendar["date"] = pd.to_datetime(
        calendar["date"]
    )

    calendar["available_numeric"] = (
        calendar["available"].map({
            "t": 1,
            "f": 0
        })
    )

    reviews["date"] = pd.to_datetime(
        reviews["date"]
    )

    return listings, calendar, reviews


listings, calendar, reviews = load_data()

st.title("Valencia Airbnb Intelligence Lab")

st.caption(
    "Maestría en Ciencia de Datos | "
    "PB - Aprendizaje de Máquina | "
    "Mat. Héctor Román"
)

st.sidebar.header("Configuración")

barrios = sorted(
    listings["neighbourhood_cleansed"]
    .dropna()
    .unique()
)

barrio = st.sidebar.selectbox(
    "Seleccionar barrio",
    ["Toda Valencia"] + list(barrios)
)

if barrio == "Toda Valencia":
    selected = listings.copy()
else:
    selected = listings[
        listings["neighbourhood_cleansed"] == barrio
    ].copy()

ids = selected["id"]

cal = calendar[
    calendar["listing_id"].isin(ids)
].copy()

rev = reviews[
    reviews["listing_id"].isin(ids)
].copy()

st.subheader(barrio)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Alojamientos",
    f"{len(selected):,}"
)

c2.metric(
    "Anfitriones",
    f"{selected['host_id'].nunique():,}"
)

c3.metric(
    "Precio mediano",
    f"{selected['price_numeric'].median():,.2f}"
)

c4.metric(
    "Disponibilidad",
    f"{cal['available_numeric'].mean()*100:.2f}%"
)

c5.metric(
    "Reseñas únicas",
    f"{rev['id'].nunique():,}"
)

tabs = st.tabs([
    "01. Exploración",
    "02. Barrios y mapas",
    "03. Precios",
    "04. Series temporales",
    "05. Consultas rápidas",
    "06. Laboratorio matemático",
    "07. Comparador de barrios",
    "08. Estancias completas",
    "09. Prediccion de demanda"
])

# ======================================
# 01. EXPLORACION
# ======================================

with tabs[0]:

    st.header("Exploración de datos")

    summary = pd.DataFrame({
        "Base": [
            "Listings",
            "Calendar",
            "Reviews"
        ],
        "Registros": [
            len(listings),
            len(calendar),
            len(reviews)
        ],
        "Columnas": [90, 5, 6]
    })

    st.dataframe(
        summary,
        width="stretch",
        hide_index=True
    )

    st.subheader("Calidad de datos")

    missing = (
        listings.isna().mean() * 100
    ).sort_values(ascending=False)

    missing = (
        missing[missing > 0]
        .reset_index()
    )

    missing.columns = [
        "Variable",
        "Porcentaje de nulos"
    ]

    fig = px.bar(
        missing.head(25),
        x="Porcentaje de nulos",
        y="Variable",
        orientation="h",
        title="Variables con mayor porcentaje de nulos"
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.subheader("Validación de reseñas")

    a, b, c = st.columns(3)

    a.metric(
        "Asociaciones",
        f"{len(reviews):,}"
    )

    b.metric(
        "IDs únicos",
        f"{reviews['id'].nunique():,}"
    )

    c.metric(
        "Pares repetidos",
        reviews.duplicated(
            ["listing_id", "id"]
        ).sum()
    )

    st.info(
        "Una misma reseña puede estar asociada "
        "a varios alojamientos. El conteo depende "
        "de la unidad de análisis."
    )

# ======================================
# 02. BARRIOS
# ======================================

with tabs[1]:

    st.header("Análisis geográfico")

    from geographic import render_map

    render_map(selected, barrio)

    counts = (
        listings["neighbourhood_cleansed"]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "Barrio",
        "Alojamientos"
    ]

    fig = px.bar(
        counts.head(20),
        x="Alojamientos",
        y="Barrio",
        orientation="h",
        title="20 barrios con más alojamientos"
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

# ======================================
# 03. PRECIOS
# ======================================

with tabs[2]:

    st.header("Análisis de precios")

    st.caption(
        "Valores monetarios según el archivo original. "
        "No representan necesariamente precios pagados."
    )

    valid = selected.dropna(
        subset=["price_numeric"]
    )

    st.dataframe(
        valid["price_numeric"]
        .describe()
        .round(2)
        .rename("Valor")
    )

    limit = st.number_input(
        "Límite superior para visualizar precios",
        min_value=10,
        max_value=50000,
        value=500,
        step=50
    )

    visible = valid[
        valid["price_numeric"] <= limit
    ]

    st.caption(
        f"Se visualizan {len(visible):,} de "
        f"{len(valid):,} precios válidos."
    )

    left, right = st.columns(2)

    with left:

        fig = px.histogram(
            visible,
            x="price_numeric",
            color="room_type",
            nbins=40,
            title="Distribución de precios"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    with right:

        fig = px.box(
            visible,
            x="room_type",
            y="price_numeric",
            color="room_type",
            title="Precios por tipo de alojamiento"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    st.subheader("Alojamientos más caros")

    top = valid.nlargest(
        10,
        "price_numeric"
    )

    st.dataframe(
        top[[
            "name",
            "price_numeric",
            "room_type",
            "listing_url"
        ]],
        column_config={
            "listing_url":
                st.column_config.LinkColumn(
                    "Abrir anuncio"
                )
        },
        hide_index=True,
        width="stretch"
    )

# ======================================
# 04. SERIES TEMPORALES
# ======================================

with tabs[3]:

    st.header("Series temporales")

    st.subheader("Disponibilidad anunciada")

    daily = (
        cal.groupby("date")
        .agg(
            disponibles=(
                "available_numeric", "sum"
            ),
            total=(
                "available_numeric", "count"
            )
        )
        .reset_index()
    )

    daily["porcentaje"] = (
        daily["disponibles"]
        / daily["total"] * 100
    )

    fig = px.line(
        daily,
        x="date",
        y="porcentaje",
        title="Disponibilidad diaria (%)"
    )

    fig.update_yaxes(range=[0, 100])

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.caption(
        "La falta de disponibilidad no equivale "
        "necesariamente a una reserva."
    )

    st.subheader("Historial de reseñas")

    frequency = st.selectbox(
        "Frecuencia temporal",
        ["Mensual", "Semanal", "Diaria"]
    )

    freq = {
        "Mensual": "MS",
        "Semanal": "W",
        "Diaria": "D"
    }[frequency]

    metric = st.radio(
        "Unidad de conteo",
        [
            "Reseñas únicas",
            "Asociaciones alojamiento-reseña"
        ],
        horizontal=True
    )

    aggregation = (
        "nunique"
        if metric == "Reseñas únicas"
        else "size"
    )

    temporal = (
        rev.groupby(
            pd.Grouper(
                key="date",
                freq=freq
            )
        )["id"]
        .agg(aggregation)
        .reset_index(name="resenas")
    )

    fig = px.line(
        temporal,
        x="date",
        y="resenas",
        title="Evolución histórica de reseñas"
    )

    fig.update_xaxes(
        rangeslider_visible=True
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.warning(
        "Las reseñas son un indicador indirecto "
        "de actividad. Los periodos inicial y "
        "final pueden estar incompletos."
    )

    st.download_button(
        "Exportar serie temporal CSV",
        temporal.to_csv(index=False),
        file_name="serie_temporal.csv",
        mime="text/csv"
    )

# ======================================
# 05. CONSULTAS RAPIDAS
# ======================================

with tabs[4]:

    st.header("Motor de consultas")

    fecha = st.date_input(
        "Fecha de consulta",
        value=cal["date"].min().date(),
        min_value=cal["date"].min().date(),
        max_value=cal["date"].max().date()
    )

    presupuesto = st.number_input(
        "Precio máximo registrado",
        min_value=0,
        max_value=50000,
        value=500,
        step=25
    )

    personas = st.slider(
        "Capacidad mínima",
        min_value=1,
        max_value=15,
        value=2
    )

    available_ids = cal.loc[
        (
            cal["date"]
            == pd.Timestamp(fecha)
        )
        & (cal["available_numeric"] == 1),
        "listing_id"
    ]

    result = selected[
        selected["id"].isin(available_ids)
        & (
            selected["price_numeric"]
            <= presupuesto
        )
        & (
            selected["accommodates"]
            >= personas
        )
    ]

    st.metric(
        "Alojamientos compatibles",
        len(result)
    )

    st.caption(
        "Consulta basada en el calendario recopilado "
        "en septiembre de 2026 y el precio general "
        "registrado en Listings. No verifica reservas "
        "ni precios en tiempo real."
    )

    st.dataframe(
        result[[
            "name",
            "room_type",
            "price_numeric",
            "accommodates",
            "listing_url"
        ]].sort_values("price_numeric"),
        column_config={
            "listing_url":
                st.column_config.LinkColumn(
                    "Ver alojamiento"
                )
        },
        hide_index=True,
        width="stretch"
    )

    st.download_button(
        "Exportar resultados",
        result[[
            "id",
            "name",
            "price_numeric",
            "listing_url"
        ]].to_csv(index=False),
        file_name="consulta.csv",
        mime="text/csv"
    )

st.sidebar.divider()

st.sidebar.markdown(
    "**Fuente:** Inside Airbnb"
)

st.sidebar.markdown(
    "[Datos originales](https://insideairbnb.com/get-the-data/)"
)

st.sidebar.caption(
    "Proyecto académico de análisis "
    "exploratorio y temporal."
)



# VERSION 2.0

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advanced_tabs import render_advanced_tabs

render_advanced_tabs(
    tabs,
    listings,
    calendar,
    reviews,
    selected,
    cal,
    rev
)


# VALENCIA PREDICTION MODULE V3
with tabs[8]:
    from prediction_tab import render_prediction_tab
    render_prediction_tab()
