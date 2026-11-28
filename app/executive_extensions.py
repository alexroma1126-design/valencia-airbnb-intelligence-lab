from pathlib import Path
import json
import unicodedata
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.forecasting import (
    CONFIG,
    load_series,
    forecast_at
)

from src.uncertainty import predict_with_uncertainty

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "app_data"

BLUE = "#38BDF8"
NAVY = "#111827"
GREEN = "#34D399"


def number(value, decimals=0):

    if value is None or not np.isfinite(value):
        return "No disponible"

    return (
        f"{value:,.{decimals}f}"
        .replace(",", "_")
        .replace(".", ",")
        .replace("_", ".")
    )


def normalize(value):

    text = unicodedata.normalize(
        "NFKD", str(value).upper()
    )

    return "".join(
        c for c in text if c.isalnum()
    )


# ==================================================
# TERRITORIAL INTELLIGENCE ENGINE
# ==================================================

@st.cache_data(show_spinner=False, max_entries=24)
def territorial_data(frequency, target, future_mode):

    target = pd.Timestamp(target)

    data = pd.read_parquet(
        DATA / CONFIG[frequency]["file"]
    )

    market = pd.read_parquet(
        DATA / "market_context.parquet"
    )

    zones = sorted(
        set(market["barrio"]) - {"TODOS"}
    )

    available = set(data["barrio"])

    rows = []
    failures = []

    saved_target = (
        "2026-11-01"
        if frequency == "mensual"
        else "2026-11-02"
    )

    # Use already validated results for November.

    if (
        future_mode
        and frequency != "diaria"
        and target == pd.Timestamp(saved_target)
    ):

        final = pd.read_csv(
            ROOT / "results" / "tables"
            / "forecast_final_predictions.csv"
        )

        final = final[
            (final["frecuencia"] == frequency)
            & (
                pd.to_datetime(final["objetivo"])
                == target
            )
            & (final["barrio"] != "TODOS")
        ]

        for row in final.to_dict("records"):

            rows.append({
                "Barrio": row["barrio"],
                "Resenas": row["prediccion"],
                "Inferior": row["inferior_95"],
                "Superior": row["superior_95"],
                "Modelo": row["modelo"],
                "WAPE": row["WAPE_test"],
                "Cobertura": (
                    100 * row["cobertura_intervalo_test"]
                )
            })

        excluded = set(zones) - set(final["barrio"])

        skipped = pd.read_csv(
            ROOT / "results" / "tables"
            / "forecast_uncertainty_skipped.csv"
        )

        skipped = skipped[
            skipped["frecuencia"] == frequency
        ]

        reasons = dict(
            zip(skipped["barrio"], skipped["motivo"])
        )

        failures = []

        for zone in sorted(excluded):

            reason = reasons.get(
                zone,
                "Sin serie temporal disponible"
                if zone not in available
                else "Sin pronostico final disponible"
            )

            failures.append(f"{zone}: {reason}")

        return pd.DataFrame(rows), failures

    # Calculate other periods using the same engine.

    for zone in zones:

        if zone not in available:

            failures.append(
                f"{zone}: sin serie disponible"
            )

            continue

        try:

            if future_mode:

                y = load_series(zone, frequency)

                result, _ = predict_with_uncertainty(
                    y=y,
                    target=target,
                    frequency=frequency
                )

                rows.append({
                    "Barrio": zone,
                    "Resenas": result["prediccion"],
                    "Inferior": result["inferior_95"],
                    "Superior": result["superior_95"],
                    "Modelo": result["modelo"],
                    "WAPE": result["WAPE_test"],
                    "Cobertura": (
                        100 *
                        result["cobertura_intervalo_test"]
                    )
                })

            else:

                observed = data[
                    (data["barrio"] == zone)
                    & (data["fecha"] == target)
                ]

                if observed.empty:
                    raise ValueError(
                        "Periodo sin observaciones"
                    )

                rows.append({
                    "Barrio": zone,
                    "Resenas": float(
                        observed["resenas"].iloc[0]
                    ),
                    "Inferior": np.nan,
                    "Superior": np.nan,
                    "Modelo": "Historico",
                    "WAPE": np.nan,
                    "Cobertura": np.nan
                })

        except ValueError as exc:

            failures.append(f"{zone}: {exc}")

    return pd.DataFrame(rows), failures


# ==================================================
# GEOGRAPHIC MATCHING
# ==================================================

@st.cache_data(show_spinner=False)
def geographic_data():

    path = (
        ROOT / "app_data"
        / "neighbourhoods.geojson"
    )

    if not path.exists():

        return None, {}

    with path.open(encoding="utf-8-sig") as file:

        geo = json.load(file)

    market = pd.read_parquet(
        DATA / "market_context.parquet"
    )

    zones = {
        normalize(z): z
        for z in market["barrio"]
        if z != "TODOS"
    }

    features = geo.get("features", [])

    properties = {
        key
        for feature in features
        for key in feature.get("properties", {})
    }

    if not properties:

        return geo, {}

    def score(key):

        matched = {
            normalize(f.get("properties", {}).get(key, ""))
            for f in features
        }

        return len(matched.intersection(zones))

    best = max(
        sorted(properties),
        key=score
    )

    mapping = {}

    for i, feature in enumerate(features):

        fid = str(i)

        feature["id"] = fid

        name = feature.get(
            "properties", {}
        ).get(best, "")

        key = normalize(name)

        if key in zones:

            mapping[fid] = zones[key]

    return geo, mapping


# ==================================================
# MAP AND DYNAMIC RANKING
# ==================================================

def render_territory(
    frequency,
    target,
    future_mode,
    rho,
    guests
):

    st.header("Inteligencia territorial")

    st.caption(
        "Compara todos los barrios disponibles "
        "para el mismo periodo y escenario turistico."
    )

    enabled = st.checkbox(
        "Calcular mapa y ranking territorial",
        key="exec_territory_enabled"
    )

    if not enabled:

        st.info(
            "Activa el analisis territorial para visualizar "
            "el mapa y el ranking de Valencia."
        )

        return

    with st.spinner(
        "Preparando resultados territoriales..."
    ):

        df, failures = territorial_data(
            frequency,
            str(pd.Timestamp(target).date()),
            future_mode
        )

    if df.empty:

        st.warning(
            "No hay resultados para este periodo."
        )

        return

    df = df.copy()

    df["Estancias"] = df["Resenas"] / rho

    df["Llegadas"] = df["Estancias"] * guests

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Barrios con resultados",
        len(df)
    )

    c2.metric(
        "Barrios sin resultados",
        len(failures)
    )

    c3.metric(
        "Barrio con mayor actividad",
        df.loc[df["Resenas"].idxmax(), "Barrio"]
    )

    metric = st.radio(
        "Indicador territorial",
        ["Resenas", "Llegadas", "Estancias"],
        horizontal=True,
        format_func=lambda x: {
            "Resenas": "Reseñas",
            "Llegadas": "Llegadas estimadas",
            "Estancias": "Estancias estimadas"
        }[x],
        key="exec_territory_metric"
    )

    geo, mapping = geographic_data()

    st.subheader("Mapa interactivo de Valencia")

    if geo is not None and mapping:

        locations = pd.DataFrame([
            {
                "id": fid,
                "Barrio": name
            }
            for fid, name in mapping.items()
        ])

        mapped = locations.merge(
            df,
            on="Barrio",
            how="left"
        )

        valid = mapped[
            mapped[metric].notna()
        ]

        missing = mapped[
            mapped[metric].isna()
        ]["id"].tolist()

        missing += [
            f["id"]
            for f in geo["features"]
            if f["id"] not in mapping
        ]

        modern = hasattr(go, "Choroplethmap")

        Trace = (
            go.Choroplethmap
            if modern
            else go.Choroplethmapbox
        )

        fig = go.Figure()

        if missing:

            fig.add_trace(Trace(
                geojson=geo,
                locations=missing,
                z=[0] * len(missing),
                colorscale=[
                    [0, "#94A3B8"],
                    [1, "#94A3B8"]
                ],
                showscale=False,
                name="Sin datos",
                text=[
                    mapping.get(
                        fid,
                        "Zona sin correspondencia"
                    )
                    for fid in missing
                ],
                customdata=[
                    dict(
                        msg.split(": ", 1)
                        for msg in failures
                        if ": " in msg
                    ).get(
                        mapping.get(fid, ""),
                        "Sin resultado disponible"
                    )
                    for fid in missing
                ],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Pronostico no disponible<br>"
                    "%{customdata}"
                    "<extra></extra>"
                ),
                marker_opacity=0.55,
                marker_line_width=0.8
            ))

        if not valid.empty:

            fig.add_trace(Trace(
                geojson=geo,
                locations=valid["id"],
                z=valid[metric],
                text=valid["Barrio"],
                customdata=valid[
                    ["Resenas", "Llegadas", "Estancias"]
                ].to_numpy(),
                colorscale=[
                    [0, "#DBEAFE"],
                    [0.35, "#60A5FA"],
                    [0.70, "#2563EB"],
                    [1, "#172554"]
                ],
                marker_opacity=0.88,
                marker_line_width=0.8,
                colorbar=dict(title=metric),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Reseñas: %{customdata[0]:,.0f}<br>"
                    "Llegadas: %{customdata[1]:,.0f}<br>"
                    "Estancias: %{customdata[2]:,.0f}"
                    "<extra></extra>"
                )
            ))

        map_config = dict(
            style="white-bg",
            center=dict(lat=39.46, lon=-0.38),
            zoom=10.5
        )

        fig.update_layout(
            **{
                "map" if modern else "mapbox":
                    map_config
            }
        )

        fig.update_layout(
            height=580,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor=NAVY,
            font=dict(color="#E2E8F0")
        )

        st.plotly_chart(
            fig,
            width="stretch",
            theme=None,
            key="exec_territory_map"
        )

        missing_names = sorted(
            set(df["Barrio"]) - set(mapping.values())
        )

        st.caption(
            f"Barrios con resultados representados: "
            f"{valid['Barrio'].nunique()} de {len(df)}. "
            "El gris representa zonas sin resultado "
            "o sin correspondencia geografica."
        )

        if missing_names:

            with st.expander(
                "Barrios sin correspondencia geografica"
            ):

                st.write(missing_names)

    else:

        st.warning(
            "No se encontraron limites geograficos "
            "compatibles. El ranking sigue disponible."
        )

    st.subheader("Ranking territorial")

    top = st.slider(
        "Numero de barrios que deseas visualizar",
        1,
        min(30, len(df)) if len(df) > 1 else 2,
        min(10, len(df)),
        key="exec_territory_top"
    )

    ranking = df.sort_values(
        metric,
        ascending=False
    ).head(top)

    ranking = ranking.sort_values(metric)

    fig = go.Figure(go.Bar(
        x=ranking[metric],
        y=ranking["Barrio"],
        orientation="h",
        marker=dict(
            color=ranking[metric],
            colorscale=[
                [0, "#1D4ED8"],
                [1, "#38BDF8"]
            ]
        ),
        text=ranking[metric].map(number),
        textposition="auto"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=NAVY,
        plot_bgcolor=NAVY,
        height=max(400, 32 * len(ranking) + 100),
        margin=dict(l=20, r=30, t=30, b=40),
        xaxis_title=metric,
        font=dict(color="#E2E8F0")
    )

    st.plotly_chart(
        fig,
        width="stretch",
        theme=None,
        key="exec_territory_ranking"
    )

    st.dataframe(
        df.sort_values(
            metric, ascending=False
        ).round(2),
        hide_index=True,
        width="stretch"
    )

    if failures:

        with st.expander(
            f"Consultar {len(failures)} exclusiones"
        ):

            for message in failures:

                st.write(message)

    st.caption(
        "El ranking mide actividad basada en resenas, "
        "no rentabilidad ni ocupacion real. "
        "Toda Valencia se calcula independientemente "
        "y no se obtiene sumando los barrios."
    )

    st.download_button(
        "Descargar analisis territorial",
        df.to_csv(index=False).encode("utf-8-sig"),
        file_name="valencia_analisis_territorial.csv",
        mime="text/csv",
        key="exec_territory_export"
    )


# ==================================================
# MATHEMATICAL EXPLAINABILITY
# ==================================================

def render_math(
    barrio,
    frequency,
    target,
    future_mode,
    y,
    summary,
    value,
    rho,
    guests
):

    st.header("Laboratorio matemático interactivo")

    if future_mode:

        st.subheader("1. Construccion del pronostico")

        season = CONFIG[frequency]["season"]

        windows = {
            "diaria": [7, 28, 91, 364],
            "semanal": [4, 13, 26, 52],
            "mensual": [3, 6, 12]
        }[frequency]

        selected = summary["modelo"]

        choices = ["A"] + [
            f"B_K{k}" for k in windows
        ]

        model = st.selectbox(
            "Modelo que deseas estudiar",
            choices,
            index=choices.index(selected),
            key=f"exec_math_model_{frequency}_{selected}"
        )

        k = (
            season
            if model == "A"
            else int(model.split("K")[-1])
        )

        result = forecast_at(
            y=y,
            target=target,
            frequency=frequency,
            k=k
        )

        base = result["base_estacional"]

        a, b, c = st.columns(3)

        a.metric("Estacionalidad", season)

        b.metric(
            "Horizonte",
            result["horizonte"]
        )

        c.metric(
            "Base estacional",
            number(base)
        )

        st.write(
            "Fecha de referencia:",
            result["fecha_referencia"].strftime(
                "%d/%m/%Y"
            )
        )

        if model == "A":

            st.latex(
                r"\widehat{Y}_{t+h}=Y_{t+h-s}"
            )

            st.latex(
                rf"\widehat{{Y}}={base:.0f}"
            )

        else:

            st.latex(
                r"g_t="
                r"\frac{\sum_{j=0}^{K-1}Y_{t-j}}"
                r"{\sum_{j=0}^{K-1}Y_{t-s-j}}"
            )

            recent = result["volumen_reciente"]

            previous = result["volumen_anterior"]

            growth = result["factor_crecimiento"]

            a, b, c = st.columns(3)

            a.metric(
                "Ventana K",
                k
            )

            b.metric(
                "Volumen reciente",
                number(recent)
            )

            c.metric(
                "Volumen anterior",
                number(previous)
            )

            if np.isfinite(growth):

                st.latex(
                    rf"g_t="
                    rf"\frac{{{recent:.0f}}}"
                    rf"{{{previous:.0f}}}"
                    rf"\approx {growth:.6f}"
                )

                st.latex(
                    rf"\widehat{{Y}}="
                    rf"{base:.0f}"
                    rf"\times {growth:.6f}"
                    rf"\approx {result['modelo_B']:.2f}"
                )

            else:

                st.warning(
                    "El modelo B no dispone de una "
                    "ventana historica valida."
                )

        rule_value = (
            result["modelo_A"]
            if model == "A"
            else result["modelo_B"]
        )

        st.metric(
            "Pronostico de la regla estudiada",
            number(rule_value, 2)
        )

        st.caption(
            f"Modelo oficial seleccionado: {selected}. "
            "Cambiar el modelo de estudio no modifica "
            "la respuesta ejecutiva ni su intervalo."
        )

        st.divider()

        st.subheader("2. Incertidumbre estadistica")

        n = summary["n_calibracion"]

        rank = int(np.ceil((n+1)*0.95))

        radius = summary["radio_intervalo"]

        st.latex(
            r"k_{\alpha}="
            r"\left\lceil(n+1)(1-\alpha)\right\rceil"
        )

        st.latex(
            rf"k_{{0.05}}="
            rf"\left\lceil({n}+1)(0.95)\right\rceil"
            rf"={rank}"
        )

        if np.isfinite(radius):

            st.latex(
                r"I=\left["
                r"\max(0,\lfloor\widehat Y-q\rfloor),"
                r"\lceil\widehat Y+q\rceil\right]"
            )

            st.latex(
                rf"q={radius:.2f}"
            )

            st.latex(
                rf"I=[{summary['inferior_95']:.0f},"
                rf"{summary['superior_95']:.0f}]"
            )

            st.caption(
                "Este intervalo corresponde al modelo "
                "oficial seleccionado, no a la regla "
                "alternativa de estudio."
            )

        else:

            st.warning(
                "Calibracion insuficiente para el intervalo."
            )

    else:

        st.info(
            "En una consulta historica no se necesita "
            "pronosticar: las resenas son observadas."
        )

    st.divider()

    st.subheader("Conversion de resenas a llegadas")

    st.latex(
        r"E=\frac{R}{\rho}"
    )

    st.latex(
        rf"E=\frac{{{value:.2f}}}"
        rf"{{{rho:.2f}}}"
        rf"\approx {value/rho:.2f}"
    )

    st.latex(
        r"P=E\cdot G"
    )

    st.latex(
        rf"P={value/rho:.2f}"
        rf"\times {guests:.2f}"
        rf"\approx {value*guests/rho:.2f}"
    )

    st.metric(
        "Llegadas estimadas",
        number(value * guests / rho)
    )

    st.caption(
        "Los resultados se calculan sin redondear "
        "los valores intermedios."
    )


# ==================================================
# EXECUTIVE INTERPRETATION
# ==================================================

def render_report(
    barrio,
    period,
    future_mode,
    summary,
    value,
    rho,
    guests
):

    name = (
        "Toda Valencia"
        if barrio == "TODOS"
        else barrio
    )

    mode = (
        "Pronostico"
        if future_mode
        else "Consulta historica"
    )

    text = [
        "VALENCIA INTELLIGENCE STUDIO",
        "RESUMEN EJECUTIVO",
        "",
        f"Ambito: {name}",
        f"Periodo: {period}",
        f"Consulta: {mode}",
        "",
        f"Resenas: {number(value)}",
        f"Estancias estimadas: {number(value/rho)}",
        f"Llegadas estimadas: {number(value*guests/rho)}",
        "",
        f"Supuesto de resenas: {rho:.0%}",
        f"Huespedes por estancia: {guests:.1f}"
    ]

    if future_mode:

        text += [
            "",
            "EVALUACION DEL MODELO",
            f"Modelo: {summary['modelo']}",
            f"WAPE: {number(summary['WAPE_test'], 2)}%",
            f"Periodos de test: {summary['n_test']}",
            "Cobertura historica: " + number(
                100 * summary["cobertura_intervalo_test"],
                1
            ) + "%"
        ]

        lower = summary["inferior_95"]
        upper = summary["superior_95"]

        if np.isfinite(lower) and np.isfinite(upper):

            text += [
                "Intervalo nominal 95% de resenas: "
                f"{number(lower)} - {number(upper)}",
                "Rango condicionado de llegadas: "
                f"{number(lower*guests/rho)} - "
                f"{number(upper*guests/rho)}"
            ]

        if summary["cobertura_intervalo_test"] < 0.95:

            text.append(
                "Precaucion: cobertura historica inferior al 95%."
            )

    text += [
        "",
        "INTERPRETACION",
        "Los resultados apoyan la planificacion turistica.",
        "Las llegadas son estimaciones, no personas observadas.",
        "La fecha de resena puede diferir de la estancia.",
        "El intervalo no garantiza cobertura futura.",
        "No se demuestra ocupacion ni rentabilidad.",
        "",
        "Fuente: Inside Airbnb.",
        "Motor cientifico: v3.2.0.",
        "Autor: Mat. Hector Roman."
    ]

    report = "\n".join(text)

    st.subheader("Resumen ejecutivo automático")

    st.text(report)

    st.download_button(
        "Descargar resumen ejecutivo",
        report.encode("utf-8-sig"),
        file_name="valencia_resumen_ejecutivo.txt",
        mime="text/plain",
        key="exec_report_download"
    )
