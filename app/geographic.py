from pathlib import Path
import html
import json

import pandas as pd
import folium
import streamlit as st

from folium.plugins import MarkerCluster, Fullscreen
from streamlit_folium import st_folium
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]


def render_map(selected, barrio):

    data = selected.copy()

    data["latitude"] = pd.to_numeric(
        data["latitude"], errors="coerce"
    )

    data["longitude"] = pd.to_numeric(
        data["longitude"], errors="coerce"
    )

    data = data.dropna(
        subset=["latitude", "longitude"]
    )

    data = data[
        data["latitude"].between(-90, 90)
        & data["longitude"].between(-180, 180)
    ]

    if data.empty:
        st.warning("No existen coordenadas validas.")
        return

    st.caption(
        f"Coordenadas validas: {len(data):,} alojamientos"
    )

    center = [
        float(data["latitude"].median()),
        float(data["longitude"].median())
    ]

    colors = {
        "Entire home/apt": "blue",
        "Private room": "green",
        "Shared room": "orange",
        "Hotel room": "red"
    }

    mode = st.radio(
        "Modo de visualizacion",
        [
            "Mapa interactivo",
            "Coordenadas sin mapa externo"
        ],
        horizontal=True,
        key="geographic_mode"
    )

    if mode == "Coordenadas sin mapa externo":

        fig = px.scatter(
            data,
            x="longitude",
            y="latitude",
            color="room_type",
            hover_name="name",
            hover_data=["price_numeric"],
            title="Distribucion espacial de alojamientos",
            height=650,
            render_mode="svg"
        )

        fig.update_layout(
            xaxis_title="Longitud",
            yaxis_title="Latitud"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        return

    m = folium.Map(
        location=center,
        zoom_start=11 if barrio == "Toda Valencia" else 14,
        tiles=None,
        control_scale=True
    )

    folium.TileLayer(
        "OpenStreetMap",
        name="OpenStreetMap"
    ).add_to(m)





    geojson_path = (
        ROOT / "data" / "raw"
        / "neighbourhoods.geojson"
    )

    if geojson_path.exists():

        with geojson_path.open(
            encoding="utf-8-sig"
        ) as file:
            geojson = json.load(file)

        folium.GeoJson(
            geojson,
            name="Limites de barrios",
            style_function=lambda feature: {
                "color": "#38BDF8",
                "weight": 1.5,
                "fillOpacity": 0.06
            }
        ).add_to(m)

    cluster = MarkerCluster(
        name="Alojamientos"
    ).add_to(m)

    for row in data.itertuples(index=False):

        name = html.escape(str(row.name))
        room = html.escape(str(row.room_type))

        price = (
            f"{row.price_numeric:,.2f}"
            if pd.notna(row.price_numeric)
            else "No disponible"
        )

        url = html.escape(
            str(row.listing_url),
            quote=True
        )

        popup = (
            f"<b>{name}</b><br>"
            f"Tipo: {room}<br>"
            f"Precio registrado: {price}<br>"
            f'<a href="{url}" target="_blank" '
            f'rel="noopener noreferrer">'
            f"Ver anuncio</a>"
        )

        folium.Marker(
            location=[
                float(row.latitude),
                float(row.longitude)
            ],
            tooltip=name,
            popup=folium.Popup(
                popup,
                max_width=300
            ),
            icon=folium.Icon(
                color=colors.get(
                    row.room_type, "gray"
                ),
                icon="home"
            )
        ).add_to(cluster)

    if len(data) > 1:

        m.fit_bounds([
            [
                float(data["latitude"].min()),
                float(data["longitude"].min())
            ],
            [
                float(data["latitude"].max()),
                float(data["longitude"].max())
            ]
        ])

    Fullscreen().add_to(m)

    folium.LayerControl().add_to(m)

    st_folium(
        m,
        height=650,
        use_container_width=True,
        returned_objects=[],
        key=f"folium_map_{barrio}"
    )

    st.caption(
        "Fuente: Inside Airbnb. "
        "Las coordenadas de los anuncios "
        "pueden ser aproximadas."
    )
