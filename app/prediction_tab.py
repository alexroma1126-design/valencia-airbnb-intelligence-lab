from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.forecasting import (
    CONFIG,
    load_series,
    forecast_at
)

from src.uncertainty import (
    predict_with_uncertainty
)

from src.tourism import (
    estimate_demand,
    period_context
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "app_data"
OUT = ROOT / "results" / "tables"


@st.cache_data(show_spinner=False)
def load_context():

    market = pd.read_parquet(
        DATA / "market_context.parquet"
    )

    daily = pd.read_parquet(
        DATA / "calendar_daily_context.parquet"
    )

    return market, daily


@st.cache_data(show_spinner=False)
def load_available_series(frequency):

    return pd.read_parquet(
        DATA / CONFIG[frequency]["file"]
    )


@st.cache_data(
    show_spinner="Evaluando modelos y calculando pronostico...",
    max_entries=128
)
def calculate_forecast(barrio, frequency, target):

    y = load_series(
        barrio,
        frequency
    )

    summary, detail = predict_with_uncertainty(
        y=y,
        target=target,
        frequency=frequency
    )

    return summary, detail


def fmt(value, decimals=0):

    if not np.isfinite(value):
        return "No disponible"

    return f"{value:,.{decimals}f}"


def model_name(model):

    if model == "A":
        return "A - Estacional"

    return f"{model} - Estacional con crecimiento"


def render_prediction_tab():

    st.title("Valencia | Inteligencia predictiva")

    st.caption(
        "Pronostico de actividad turistica por barrio "
        "mediante series temporales y validacion cronologica."
    )

    market, daily = load_context()

    barrios = sorted(
        market["barrio"].unique().tolist()
    )

    barrios = ["TODOS"] + [
        b for b in barrios if b != "TODOS"
    ]

    c1, c2 = st.columns([2, 1])

    with c1:

        barrio = st.selectbox(
            "Seleccionar barrio",
            barrios,
            index=barrios.index("RUSSAFA")
            if "RUSSAFA" in barrios else 0,
            format_func=lambda b: (
                "Toda Valencia"
                if b == "TODOS" else b
            ),
            key="pred_barrio"
        )

    with c2:

        frequency = st.radio(
            "Frecuencia temporal",
            ["mensual", "semanal"],
            horizontal=True,
            format_func=str.capitalize,
            key="pred_frequency"
        )

    series_data = load_available_series(
        frequency
    )

    if barrio not in set(series_data["barrio"]):

        st.warning(
            "Este barrio no dispone de una serie "
            "temporal suficiente para esta frecuencia."
        )

        return

    y = load_series(
        barrio,
        frequency
    )

    origin = y.index.max()

    cfg = CONFIG[frequency]

    dates = pd.date_range(
        start=origin,
        periods=cfg["season"] + 1,
        freq=cfg["freq"]
    )[1:]

    default_date = pd.Timestamp(
        "2026-11-01"
        if frequency == "mensual"
        else "2026-11-02"
    )

    default_index = (
        dates.get_loc(default_date)
        if default_date in dates else 0
    )

    def format_period(value):

        value = pd.Timestamp(value)

        if frequency == "mensual":

            return value.strftime("%m/%Y")

        end = value + pd.Timedelta(days=6)

        return (
            f"{value:%d/%m/%Y} - "
            f"{end:%d/%m/%Y}"
        )

    target = st.selectbox(
        "Periodo que desea predecir",
        dates.tolist(),
        index=default_index,
        format_func=format_period,
        key=f"pred_target_{frequency}"
    )

    with st.expander(
        "Supuestos para estimar estancias y huespedes",
        expanded=False
    ):

        c1, c2 = st.columns(2)

        with c1:

            rho = st.slider(
                "Porcentaje de estancias que dejan resena",
                20,
                90,
                50,
                5,
                key="pred_review_rate"
            ) / 100

        with c2:

            guests = st.slider(
                "Huespedes promedio por estancia",
                1.0,
                5.0,
                2.0,
                0.5,
                key="pred_guests"
            )

        st.caption(
            "Los parametros son supuestos de escenario, "
            "no valores calibrados con reservas reales."
        )

    st.caption(
        f"Ultimo periodo completo utilizado: "
        f"{format_period(origin)}"
    )

    try:

        summary, detail = calculate_forecast(
            barrio,
            frequency,
            str(pd.Timestamp(target).date())
        )

    except ValueError as exc:

        st.warning(
            "No es posible emitir un pronostico "
            "validado para esta configuracion."
        )

        st.info(str(exc))

        st.line_chart(
            y.rename("Resenas observadas")
        )

        return

    value = summary["prediccion"]

    lower = summary["inferior_95"]
    upper = summary["superior_95"]

    coverage = summary[
        "cobertura_intervalo_test"
    ]

    estimates = estimate_demand(
        reviews=value,
        review_rate=rho,
        guests=guests
    )

    context = period_context(
        daily,
        barrio,
        target,
        frequency
    )

    interval_available = (
        np.isfinite(lower)
        and np.isfinite(upper)
    )

    tabs = st.tabs([
        "01. Respuesta",
        "02. Pronostico",
        "03. Evaluacion",
        "04. Mercado",
        "05. Metodologia"
    ])

    # ========================================
    # 01. RESPUESTA EJECUTIVA
    # ========================================

    with tabs[0]:

        st.subheader(
            f"Resultado: {format_period(target)}"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Resenas previstas",
            fmt(value)
        )

        c2.metric(
            "Intervalo nominal 95%",
            (
                f"{fmt(lower)} - {fmt(upper)}"
                if interval_available
                else "No disponible"
            )
        )

        c3.metric(
            "Estancias estimadas",
            fmt(
                estimates["estancias_estimadas"]
            )
        )

        c4.metric(
            "Llegadas estimadas",
            fmt(
                estimates["llegadas_estimadas"]
            )
        )

        st.success(
            f"Para {barrio}, durante "
            f"{format_period(target)}, "
            f"se pronostican aproximadamente "
            f"{fmt(value)} resenas."
        )

        st.write(
            f"Con una tasa de resena del "
            f"{rho:.0%} y {guests:.1f} huespedes "
            f"por estancia, esto equivale a "
            f"{fmt(estimates['llegadas_estimadas'])} "
            f"llegadas estimadas de huespedes."
        )

        a, b, c = st.columns(3)

        a.metric(
            "Modelo seleccionado",
            summary["modelo"]
        )

        b.metric(
            "WAPE historico",
            fmt(summary["WAPE_test"], 2) + "%"
        )

        c.metric(
            "Cobertura historica",
            (
                f"{coverage:.1%}"
                if np.isfinite(coverage)
                else "No disponible"
            )
        )

        if np.isfinite(coverage) and coverage < 0.95:

            st.warning(
                "La cobertura historica es inferior "
                "al nivel nominal del 95%. "
                "Interpretar el intervalo con cautela."
            )

        if summary["cobertura_datos_test"] < 1:

            st.warning(
                "El modelo no dispone de todos "
                "los periodos de test previstos."
            )

        if not interval_available:

            st.warning(
                "No existe calibracion suficiente "
                "para un intervalo nominal del 95%."
            )

        if interval_available:

            st.subheader(
                "Sensibilidad de la estimacion turistica"
            )

            scenarios = pd.DataFrame({
                "Escenario": [
                    "Bajo",
                    "Central",
                    "Alto"
                ],
                "Resenas": [
                    lower,
                    value,
                    upper
                ],
                "Tasa de resena": [
                    0.70,
                    rho,
                    0.30
                ],
                "Huespedes por estancia": [
                    1.5,
                    guests,
                    3.0
                ]
            })

            scenarios["Estancias estimadas"] = (
                scenarios["Resenas"]
                / scenarios["Tasa de resena"]
            )

            scenarios["Llegadas estimadas"] = (
                scenarios["Estancias estimadas"]
                * scenarios["Huespedes por estancia"]
            )

            scenarios.loc[
                scenarios["Escenario"] == "Bajo",
                "Escenario"
            ] = "Referencia conservadora"

            scenarios.loc[
                scenarios["Escenario"] == "Alto",
                "Escenario"
            ] = "Referencia expansiva"

            st.dataframe(
                scenarios.round(2),
                hide_index=True,
                width="stretch"
            )

            st.caption(
                "Escenarios de referencia, no intervalos "
                "probabilisticos para personas. "
                "El escenario personalizado puede quedar "
                "fuera de las referencias."
            )

        st.download_button(
            "Exportar pronostico CSV",
            pd.DataFrame([{
                "barrio": barrio,
                **summary,
                **estimates,
                **context
            }]).to_csv(
                index=False
            ).encode("utf-8"),
            file_name="pronostico_valencia.csv",
            mime="text/csv",
            key="pred_download"
        )

    # ========================================
    # 02. PRONOSTICO
    # ========================================

    with tabs[1]:

        st.subheader(
            "Historia y horizonte predictivo"
        )

        history = y.tail(
            24 if frequency == "mensual"
            else 104
        )

        future = pd.date_range(
            origin,
            target,
            freq=cfg["freq"]
        )[1:]

        central = []

        for date in future:

            r = forecast_at(
                y=y,
                target=date,
                frequency=frequency,
                origin=origin,
                k=summary["K"]
            )

            p = (
                r["modelo_A"]
                if summary["modelo"] == "A"
                else r["modelo_B"]
            )

            central.append(p)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=history.index,
            y=history.values,
            name="Resenas observadas",
            mode="lines",
            line=dict(
                color="#64748b",
                width=2
            )
        ))

        fig.add_trace(go.Scatter(
            x=[origin] + future.tolist(),
            y=[float(y.iloc[-1])] + central,
            name="Pronostico ilustrativo",
            mode="lines+markers",
            line=dict(
                color="#2563eb",
                dash="dash",
                width=3
            )
        ))

        if interval_available:

            fig.add_trace(go.Scatter(
                x=[target],
                y=[value],
                name="Objetivo e intervalo 95%",
                mode="markers",
                marker=dict(
                    size=12,
                    color="#16a34a"
                ),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[upper-value],
                    arrayminus=[value-lower],
                    visible=True
                )
            ))

        fig.update_layout(
            template="plotly_white",
            height=480,
            hovermode="x unified",
            xaxis_title="Fecha",
            yaxis_title="Resenas",
            legend=dict(
                orientation="h",
                y=1.15
            )
        )

        fig.update_yaxes(
            rangemode="tozero"
        )

        st.plotly_chart(
            fig,
            width="stretch",
            key="pred_forecast_chart"
        )

        st.caption(
            "El intervalo corresponde exclusivamente "
            "al horizonte seleccionado. Los puntos "
            "intermedios muestran la trayectoria del "
            "mismo modelo desde un origen fijo."
        )

        st.subheader(
            "Comparacion de reglas predictivas"
        )

        comparison = forecast_at(
            y=y,
            target=target,
            frequency=frequency,
            k=summary["K"]
        )

        comparison_df = pd.DataFrame({
            "Modelo": [
                "A - Estacional",
                "B - Crecimiento ajustado"
            ],
            "Prediccion": [
                comparison["modelo_A"],
                comparison["modelo_B"]
            ]
        })

        st.bar_chart(
            comparison_df.set_index("Modelo")
        )

        st.write(
            "Modelo elegido:",
            model_name(summary["modelo"])
        )

        st.write(
            "Horizonte:",
            summary["horizonte"]
        )

        st.write(
            "Ventana K:",
            summary["K"]
        )

    # ========================================
    # 03. EVALUACION
    # ========================================

    with tabs[2]:

        st.subheader(
            "Evaluacion fuera de la seleccion"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "MAE",
            fmt(summary["MAE_test"], 2)
        )

        c2.metric(
            "RMSE",
            fmt(summary["RMSE_test"], 2)
        )

        c3.metric(
            "WAPE",
            fmt(summary["WAPE_test"], 2) + "%"
        )

        c4.metric(
            "Periodos de test",
            summary["n_test"]
        )

        test = detail["test"].copy()

        if not test.empty:

            fig = go.Figure()

            if test["inferior"].notna().all():

                fig.add_trace(go.Scatter(
                    x=test["fecha"],
                    y=test["superior"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip"
                ))

                fig.add_trace(go.Scatter(
                    x=test["fecha"],
                    y=test["inferior"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(37,99,235,0.15)",
                    line=dict(width=0),
                    name="Intervalo nominal 95%"
                ))

            fig.add_trace(go.Scatter(
                x=test["fecha"],
                y=test["real"],
                name="Observado",
                mode="lines+markers",
                line=dict(
                    color="#111827",
                    width=3
                )
            ))

            fig.add_trace(go.Scatter(
                x=test["fecha"],
                y=test["prediccion"],
                name="Modelo seleccionado",
                mode="lines+markers",
                line=dict(
                    color="#2563eb",
                    dash="dash"
                )
            ))

            fig.update_layout(
                template="plotly_white",
                height=440,
                hovermode="x unified",
                xaxis_title="Fecha",
                yaxis_title="Resenas"
            )

            st.plotly_chart(
                fig,
                width="stretch",
                key="pred_test_chart"
            )

            st.dataframe(
                test[[
                    "fecha",
                    "real",
                    "prediccion",
                    "inferior",
                    "superior",
                    "cubierto"
                ]].round({"real": 2, "prediccion": 2}),
                hide_index=True,
                width="stretch"
            )

        st.subheader(
            "Calibracion de incertidumbre"
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Observaciones de calibracion",
            summary["n_calibracion"]
        )

        c2.metric(
            "Cobertura empirica",
            (
                f"{coverage:.1%}"
                if np.isfinite(coverage)
                else "No disponible"
            )
        )

        cal = detail["calibracion"]

        if not cal.empty:

            fig = go.Figure(
                go.Histogram(
                    x=cal["error"],
                    nbinsx=20,
                    marker_color="#2563eb"
                )
            )

            fig.update_layout(
                title="Distribucion de errores historicos",
                template="plotly_white",
                xaxis_title="Observado menos predicho",
                yaxis_title="Frecuencia",
                height=350
            )

            st.plotly_chart(
                fig,
                width="stretch",
                key="pred_error_histogram"
            )

    # ========================================
    # 04. MERCADO
    # ========================================

    with tabs[3]:

        st.subheader(
            "Contexto de oferta y disponibilidad"
        )

        selected = market[
            market["barrio"] == barrio
        ]

        if not selected.empty:

            row = selected.iloc[0]

            a, b, c = st.columns(3)

            a.metric(
                "Alojamientos",
                fmt(row["alojamientos"])
            )

            b.metric(
                "Capacidad registrada",
                fmt(row["capacidad_total"])
            )

            c.metric(
                "Capacidad media",
                fmt(row["capacidad_media"], 2)
            )

        if context["calendario_completo"]:

            a, b = st.columns(2)

            a.metric(
                "Disponibilidad anunciada",
                fmt(
                    context[
                        "disponibilidad_media_pct"
                    ],
                    2
                ) + "%"
            )

            b.metric(
                "Noches-anuncio disponibles",
                fmt(
                    context[
                        "noches_anuncio_disponibles"
                    ]
                )
            )

        else:

            st.warning(
                "El calendario no cubre todo "
                "el periodo seleccionado."
            )

        data = daily[
            daily["barrio"] == barrio
        ]

        fig = go.Figure(go.Scatter(
            x=data["fecha"],
            y=data["disponibilidad_pct"],
            name="Disponibilidad",
            mode="lines",
            fill="tozeroy",
            line=dict(color="#0d9488")
        ))

        fig.update_layout(
            template="plotly_white",
            height=400,
            yaxis_title="Disponibilidad (%)",
            xaxis_title="Fecha del calendario"
        )

        fig.update_yaxes(
            range=[0, 100]
        )

        st.plotly_chart(
            fig,
            width="stretch",
            key="pred_market_chart"
        )

        st.info(
            "El calendario es una fotografia de la "
            "disponibilidad anunciada. No representa "
            "reservas ni ocupacion confirmadas."
        )

    # ========================================
    # 05. METODOLOGIA
    # ========================================

    with tabs[4]:

        st.subheader(
            "Fundamentos matematicos"
        )

        st.markdown(
            "### Variable objetivo"
        )

        st.latex(
            r"Y_{b,t}=|\{r.id:r\in R_{b,t}\}|"
        )

        st.markdown(
            "### Modelo A: naive estacional"
        )

        st.latex(
            r"\widehat{Y}_{t+h}=Y_{t+h-s}"
        )

        st.markdown(
            "### Modelo B: crecimiento interanual"
        )

        st.latex(
            r"\widehat{Y}_{t+h}=Y_{t+h-s}g_t"
        )

        st.latex(
            r"g_t=\frac{\sum_{j=0}^{K-1}Y_{t-j}}"
            r"{\sum_{j=0}^{K-1}Y_{t-s-j}}"
        )

        st.markdown(
            "### Estimacion de estancias y llegadas"
        )

        st.latex(
            r"E=\frac{R}{\rho},"
            r"\qquad P=\frac{RG}{\rho}"
        )

        st.markdown(
            "### Intervalo empirico de prediccion"
        )

        st.latex(
            r"I=[\max(0,\widehat{Y}-q),"
            r"\widehat{Y}+q]"
        )

        st.subheader(
            "Separacion temporal"
        )

        windows = detail["ventanas"]

        st.dataframe(
            pd.DataFrame([
                {
                    "Etapa": "Validacion",
                    "Inicio": windows["validation_start"],
                    "Fin": windows["validation_end"]
                },
                {
                    "Etapa": "Calibracion",
                    "Inicio": windows["calibration_start"],
                    "Fin": windows["calibration_end"]
                },
                {
                    "Etapa": "Test",
                    "Inicio": windows["test_start"],
                    "Fin": windows["test_end"]
                }
            ]),
            hide_index=True
        )

        st.markdown(
            """
### Limitaciones

- Las resenas no equivalen a reservas.
- Las llegadas no representan personas unicas.
- Los parametros turisticos son supuestos.
- El nivel nominal del 95% no garantiza cobertura futura.
- Las series pueden presentar cambios estructurales.
- La recopilacion de 2026 puede introducir sesgo de supervivencia.
- El calendario no demuestra ocupacion efectiva.
- No se utilizan variables meteorologicas o eventos externos.
            """
        )

    st.divider()

    st.caption(
        "Mat. Hector Roman | Maestria en Ciencia de Datos | "
        "Valencia Airbnb Intelligence Lab 3.0 | Inside Airbnb"
    )
