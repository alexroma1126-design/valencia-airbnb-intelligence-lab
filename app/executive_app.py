from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.forecasting import CONFIG, load_series
from src.uncertainty import predict_with_uncertainty
from src.backtesting import SETTINGS, evaluate_candidates
from src.tourism import estimate_demand, period_context


from executive_extensions import (
    render_territory,
    render_math,
    render_report
)

DATA = ROOT / "app_data"

NAVY = "#0B172A"
BLUE = "#2563EB"
CYAN = "#38BDF8"
GREEN = "#16765D"
GRAY = "#64748B"

MONTHS = [
    "Enero", "Febrero", "Marzo", "Abril",
    "Mayo", "Junio", "Julio", "Agosto",
    "Septiembre", "Octubre", "Noviembre",
    "Diciembre"
]


# ==================================================
# DATA SERVICES
# ==================================================

@st.cache_data(show_spinner=False)
def market_data():

    return (
        pd.read_parquet(DATA / "market_context.parquet"),
        pd.read_parquet(DATA / "calendar_daily_context.parquet")
    )


@st.cache_data(show_spinner=False)
def series_data(barrio, frequency):

    return load_series(barrio, frequency)


@st.cache_data(show_spinner=False)
def available_scopes(frequency):

    df = pd.read_parquet(
        DATA / CONFIG[frequency]["file"]
    )

    return sorted(df["barrio"].unique().tolist())


@st.cache_data(
    show_spinner="Calculando pronostico validado...",
    max_entries=128
)
def predict(barrio, frequency, target):

    return predict_with_uncertainty(
        y=load_series(barrio, frequency),
        target=target,
        frequency=frequency
    )


@st.cache_data(show_spinner=False, max_entries=128)
def candidates(barrio, frequency, horizon, origin,
               validation_start, validation_end):

    y = load_series(barrio, frequency).loc[:origin]

    return evaluate_candidates(
        y=y,
        frequency=frequency,
        start=validation_start,
        end=validation_end,
        horizon=horizon,
        k_values=SETTINGS[frequency]["k_values"]
    )


# ==================================================
# PRESENTATION HELPERS
# ==================================================

def number(value, decimals=0):

    if value is None or not np.isfinite(value):
        return "No disponible"

    return (
        f"{value:,.{decimals}f}"
        .replace(",", "_")
        .replace(".", ",")
        .replace("_", ".")
    )


def percent(value, decimals=1):

    if value is None or not np.isfinite(value):
        return "No disponible"

    return number(value, decimals) + "%"


def zone_name(barrio):

    return "Toda Valencia" if barrio == "TODOS" else barrio


def period_label(date, frequency):

    date = pd.Timestamp(date)

    if frequency == "mensual":

        return f"{MONTHS[date.month-1]} de {date.year}"

    end = date + pd.Timedelta(days=6)

    return f"{date:%d/%m/%Y} - {end:%d/%m/%Y}"


def style_chart(fig, title, ytitle="Resenas", height=390):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        title=dict(text=title, x=0.02, y=0.98, xanchor="left", yanchor="top"),
        height=height,
        margin=dict(l=35, r=30, t=110, b=45),
        hovermode="x unified",
        font=dict(size=13, color="#E2E8F0"),
        legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"),
        yaxis_title=ytitle
    )

    fig.update_yaxes(rangemode="tozero", gridcolor="#334155")

    return fig


# ==================================================
# MAIN EXECUTIVE APPLICATION
# ==================================================

def main():

    st.set_page_config(
        page_title="Valencia Intelligence Studio",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown(
        """
        <div style="
            background:linear-gradient(120deg,#0B172A,#173B63);
            padding:15px 22px;
            border-radius:15px;
            color:white;
            margin-bottom:15px;">
            <div style="color:#38BDF8;font-size:13px;">
                TOURISM | BUSINESS INTELLIGENCE
            </div>
            <h1 style="color:white;margin:3px 0;font-size:30px;">
                Valencia Intelligence Studio
            </h1>
            <p style="color:#CBD5E1;margin-bottom:5px;">
                Inteligencia turística y predicción de demanda
            </p>
            <small>
                Mat. Héctor Román | Maestria en Ciencia de Datos
            </small>
        </div>
        """,
        unsafe_allow_html=True
    )


    # EXECUTIVE_DESIGN_V2

    st.markdown(
        '''
        <style>

        .block-container {
            padding-top: 3.5rem;
            padding-bottom: 1.5rem;
            max-width: 1600px;
        }

        div[data-testid="stMetric"] {
            background: rgba(100,116,139,0.09);
            border: 1px solid rgba(148,163,184,0.22);
            border-radius: 12px;
            padding: 16px;
            min-height: 110px;
        }

        div[data-testid="stMetricValue"] {
            font-weight: 650;
            font-size: 1.85rem;
        }

        div[data-testid="stMetricLabel"] {
            font-size: 0.85rem;
        }

        </style>
        ''',
        unsafe_allow_html=True
    )

    market, daily = market_data()

    with st.sidebar:
        st.header("Configura tu consulta")

        left, right = st.container(), st.container()

        with left:

            mode = st.radio(
                "Tipo de consulta",
                ["Pronostico", "Historico"],
            format_func=lambda x: "Pronóstico" if x == "Pronostico" else "Histórico",
                horizontal=True,
                key="exec_mode"
            )

        with right:

            frequency = st.radio(
                "Frecuencia temporal",
                ["mensual", "semanal"],
                horizontal=True,
                format_func=str.capitalize,
                key="exec_frequency"
            )

        future_mode = mode == "Pronostico"

        scopes = available_scopes(frequency)

        scopes = ["TODOS"] + [
            b for b in scopes if b != "TODOS"
        ]

        a, b = st.container(), st.container()

        with a:

            barrio = st.selectbox(
                "Barrio",
                scopes,
                index=scopes.index("RUSSAFA")
                if "RUSSAFA" in scopes else 0,
                format_func=zone_name,
                key=f"exec_barrio_{frequency}"
            )

        y = series_data(barrio, frequency)

        origin = y.index.max()

        cfg = CONFIG[frequency]

        if future_mode:

            dates = pd.date_range(
                origin,
                periods=cfg["season"] + 1,
                freq=cfg["freq"]
            )[1:]

            default = (
                "2026-11-01"
                if frequency == "mensual"
                else "2026-11-02"
            )

        else:

            dates = y.index[::-1]

            default = (
                "2026-03-01"
                if frequency == "mensual"
                else "2026-03-02"
            )

        options = [str(x.date()) for x in dates]

        with b:

            target = pd.Timestamp(
                st.selectbox(
                    "Período de consulta",
                    options,
                    index=options.index(default)
                    if default in options else 0,
                    format_func=lambda x: period_label(
                        x, frequency
                    ),
                    key=f"exec_period_{mode}_{frequency}_{barrio}"
                )
            )

        st.caption(
            "Fuente: Inside Airbnb | "
            f"Último período completo: "
            f"{period_label(origin, frequency)}"
        )

        # ==============================================
        # INTERACTIVE BUSINESS ASSUMPTIONS
        # ==============================================

        with st.expander(
            "Simulador de demanda turística",
            expanded=True
        ):

            a, b = st.container(), st.container()

            with a:

                rho = st.slider(
                    "Estancias que dejan reseña (%)",
                    20, 90, 50, 5,
                    key="exec_rho"
                ) / 100

            with b:

                guests = st.slider(
                    "Huéspedes promedio por estancia",
                    1.0, 5.0, 2.0, 0.5,
                    key="exec_guests"
                )

            st.caption(
                "Estos parametros son supuestos de escenario. "
                "No se han estimado con reservas observadas."
            )

        # ==============================================
        # EXECUTE QUERY
        # ==============================================


    summary = None
    detail = None

    lower = np.nan
    upper = np.nan

    if future_mode:

        try:

            summary, detail = predict(
                barrio,
                frequency,
                str(target.date())
            )

        except ValueError as exc:

            st.warning(
                "No existe un pronostico validado "
                "para esta consulta."
            )

            st.info(str(exc))

            st.line_chart(y)

            return

        value = summary["prediccion"]

        lower = summary["inferior_95"]
        upper = summary["superior_95"]

    else:

        value = float(y.loc[target])

    estimates = estimate_demand(
        reviews=value,
        review_rate=rho,
        guests=guests
    )

    context = period_context(
        daily, barrio, target, frequency
    )

    has_interval = (
        np.isfinite(lower)
        and np.isfinite(upper)
    )

    # ==============================================
    # BUSINESS ANSWER
    # ==============================================

    st.subheader("Respuesta ejecutiva")

    st.markdown(
        f"### {zone_name(barrio)} | "
        f"{period_label(target, frequency)}"
    )

    c0, c1, c2, c3 = st.columns([1.4, 1, 1, 1])
    c0.metric(
        "Llegadas estimadas",
        number(estimates["llegadas_estimadas"]),
        help=(
            "Estimacion indirecta de llegadas, "
            "no personas unicas observadas."
        )
    )

    c1.metric(
        "Reseñas previstas" if future_mode
        else "Reseñas observadas",
        number(value)
    )

    c2.metric(
        "Estancias estimadas",
        number(estimates["estancias_estimadas"])
    )

    c3.metric(
        "Disponibilidad anunciada",
        percent(context["disponibilidad_media_pct"])
    )

    if future_mode:

        a, b, c, d = st.columns(4)

        a.metric(
            "Modelo seleccionado",
            summary["modelo"]
        )

        b.metric(
            "WAPE en test",
            percent(summary["WAPE_test"], 2)
        )

        c.metric(
            "Cobertura empírica",
            percent(
                100 * summary["cobertura_intervalo_test"]
            )
        )

        d.metric(
            "Intervalo nominal 95% de reseñas",
            f"{number(lower)} - {number(upper)}"
            if has_interval else "No disponible"
        )

        if not has_interval:

            st.warning(
                "Calibracion insuficiente para construir "
                "un intervalo nominal del 95%."
            )

        if summary["cobertura_intervalo_test"] < 0.95:

            st.warning(
                "La cobertura historica es inferior al 95%. "
                "Interpretar el pronostico con cautela."
            )

        if summary["cobertura_datos_test"] < 1:

            st.warning(
                "No se dispone de todos los periodos "
                "de test previstos."
            )

    else:

        st.info(
            "Consulta historica: las resenas fueron observadas. "
            "Las estancias y llegadas siguen siendo estimaciones."
        )

    st.write(
        f"Con {number(value)} resenas, una tasa de "
        f"publicacion del {rho:.0%} y {guests:.1f} "
        f"huespedes por estancia, se estiman "
        f"{number(estimates['estancias_estimadas'])} estancias "
        f"y {number(estimates['llegadas_estimadas'])} llegadas."
    )

    if has_interval:

        st.info(
            "Rango de llegadas con supuestos fijos: "
            f"{number(lower * guests / rho)} a "
            f"{number(upper * guests / rho)}. "
            "Es una transformacion del intervalo de resenas, "
            "no un intervalo validado con huespedes reales."
        )

    # ==============================================
    # INTERACTIVE TABS
    # ==============================================

    tabs = st.tabs([
        "Evolución y pronóstico",
        "Comparar barrios",
        "Modelos y confiabilidad",
        "Simulador turístico",
        "Mercado y metodología",
        "Inteligencia territorial",
        "Laboratorio matematico"
    ])

    # ==============================================
    # 01. HISTORY AND FORECAST
    # ==============================================

    with tabs[0]:

        st.subheader("Evolución temporal")

        if future_mode:
            history = y.tail(
                36 if frequency == "mensual" else 104
            )
        else:
            history = y.loc[:target].tail(
                36 if frequency == "mensual" else 104
            )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=history.index,
            y=history.values,
            mode="lines",
            name="Historia observada",
            line=dict(color="#94A3B8", width=3)
        ))

        if future_mode:

            error = None

            if has_interval:

                error = dict(
                    type="data",
                    symmetric=False,
                    array=[upper-value],
                    arrayminus=[value-lower],
                    visible=True
                )

            fig.add_trace(go.Scatter(
                x=[target],
                y=[value],
                mode="markers",
                name="Pronóstico seleccionado",
                marker=dict(size=13, color=BLUE),
                error_y=error
            ))

            fig.add_trace(go.Scatter(
                x=[origin, target],
                y=[float(y.iloc[-1]), value],
                mode="lines",
                name="Conexión ilustrativa",
                line=dict(color=BLUE, dash="dot")
            ))

        else:

            fig.add_trace(go.Scatter(
                x=[target],
                y=[value],
                mode="markers",
                name="Periodo consultado",
                marker=dict(size=13, color=GREEN)
            ))

        fig = style_chart(
            fig, "Actividad turística basada en reseñas"
        )

        fig.update_xaxes(
            rangeslider_visible=True
        )

        st.plotly_chart(
            fig,
            width="stretch",
            theme=None, key="exec_history_chart"
        )

        st.caption(
            "Puedes acercar, desplazar y consultar "
            "valores individuales sobre la grafica. "
            "La linea punteada no representa pronosticos "
            "intermedios."
        )

    # ==============================================
    # 02. COMPARATIVE INTELLIGENCE
    # ==============================================

    with tabs[1]:

        st.subheader("Comparacion territorial")

        st.write(
            "Compara diferentes barrios utilizando "
            "el mismo periodo y los mismos supuestos."
        )

        enabled = st.checkbox(
            "Activar comparacion",
            key="exec_compare"
        )

        if enabled:

            other = [
                z for z in scopes
                if z not in [barrio, "TODOS"]
            ]

            selected = st.multiselect(
                "Barrios adicionales",
                other,
                default=[
                    z for z in [
                        "RUSSAFA",
                        "CABANYAL-CANYAMELAR",
                        "BENIFERRI"
                    ] if z in other
                ][:2],
                max_selections=3,
                key=f"exec_compare_{frequency}_{barrio}"
            )

            results = []

            failures = []

            for zone in [barrio] + selected:

                try:

                    if future_mode:

                        result, _ = predict(
                            zone, frequency, str(target.date())
                        )

                        r = result["prediccion"]

                        wape = result["WAPE_test"]

                    else:

                        sy = series_data(zone, frequency)

                        if target not in sy.index:

                            raise ValueError(
                                "Periodo no observado"
                            )

                        r = float(sy.loc[target])

                        wape = np.nan

                    e = estimate_demand(
                        r, rho, guests
                    )

                    results.append({
                        "Barrio": zone_name(zone),
                        "Resenas": r,
                        "Estancias": e["estancias_estimadas"],
                        "Llegadas": e["llegadas_estimadas"],
                        "WAPE": wape
                    })

                except ValueError as exc:

                    failures.append(
                        f"{zone_name(zone)}: {exc}"
                    )

            if results:

                df = pd.DataFrame(results).sort_values(
                    "Llegadas",
                    ascending=True
                )

                fig = go.Figure(go.Bar(
                    x=df["Llegadas"],
                    y=df["Barrio"],
                    orientation="h",
                    marker_color=BLUE,
                    text=df["Llegadas"].map(number),
                    textposition="auto"
                ))

                fig = style_chart(
                    fig,
                    "Llegadas estimadas por barrio",
                    ""
                )

                fig.update_layout(hovermode="closest")
                fig.update_xaxes(title="Llegadas estimadas")

                st.plotly_chart(
                    fig,
                    width="stretch",
                    theme=None, key="exec_compare_chart"
                )

                st.dataframe(
                    df.round(2),
                    hide_index=True,
                    width="stretch"
                )

            for failure in failures:

                st.warning(failure)

            st.caption(
                "Esta comparacion no constituye un ranking "
                "de rentabilidad ni de ocupacion real."
            )

    # ==============================================
    # 03. MODEL EXPLAINABILITY
    # ==============================================

    with tabs[2]:

        st.subheader("Por que confiamos en este modelo")

        if not future_mode:

            st.info(
                "Selecciona Pronostico para visualizar "
                "la seleccion y evaluacion de modelos."
            )

        else:

            windows = detail["ventanas"]

            ranking = candidates(
                barrio,
                frequency,
                summary["horizonte"],
                str(origin.date()),
                str(windows["validation_start"].date()),
                str(windows["validation_end"].date())
            )

            valid = ranking[
                (ranking["cobertura"] == 1)
                & ranking["MAE"].notna()
            ].copy()

            fig = go.Figure(go.Bar(
                x=valid["modelo"],
                y=valid["MAE"],
                marker_color=[
                    GREEN if m == summary["modelo"]
                    else BLUE
                    for m in valid["modelo"]
                ],
                text=valid["MAE"].round(2),
                textposition="auto"
            ))

            st.plotly_chart(
                style_chart(
                    fig,
                    "Seleccion de modelos: menor MAE",
                    "MAE de validacion",
                    350
                ),
                width="stretch",
                theme=None, key="exec_candidates"
            )

            st.success(
                f"Modelo seleccionado: {summary['modelo']}. "
                "La decision utiliza validacion cronologica, "
                "no los errores del test."
            )

            st.dataframe(
                ranking[[
                    "modelo", "MAE", "RMSE", "WAPE",
                    "cobertura"
                ]].round(3),
                hide_index=True,
                width="stretch"
            )

            st.caption(
                "Solo compiten modelos con cobertura "
                "completa en validacion."
            )

            with st.expander(
                "Entender matematicamente los modelos"
            ):

                st.write(
                    "Modelo A: repite el periodo "
                    "estacional anterior."
                )

                st.latex(
                    r"\widehat{Y}_{t+h}=Y_{t+h-s}"
                )

                st.write(
                    "Modelo B: ajusta el periodo "
                    "estacional mediante crecimiento."
                )

                st.latex(
                    r"\widehat{Y}_{t+h}=Y_{t+h-s}g_t"
                )

                st.latex(
                    r"g_t=\frac{\sum_{j=0}^{K-1}Y_{t-j}}"
                    r"{\sum_{j=0}^{K-1}Y_{t-s-j}}"
                )

                st.write(
                    f"Horizonte: {summary['horizonte']}"
                )

                if summary["modelo"] != "A":

                    st.write(
                        f"Ventana K: {summary['K']}"
                    )

            st.divider()

            st.subheader("Evaluacion en datos de test")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "MAE",
                number(summary["MAE_test"], 2)
            )

            c2.metric(
                "RMSE",
                number(summary["RMSE_test"], 2)
            )

            c3.metric(
                "Periodos de test",
                summary["n_test"]
            )

            c4.metric(
                "Errores de calibracion",
                summary["n_calibracion"]
            )

            test = detail["test"]

            fig = go.Figure()

            if not test.empty:

                if test["inferior"].notna().all():

                    fig.add_trace(go.Scatter(
                        x=test["fecha"],
                        y=test["superior"],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False
                    ))

                    fig.add_trace(go.Scatter(
                        x=test["fecha"],
                        y=test["inferior"],
                        mode="lines",
                        fill="tonexty",
                        fillcolor="rgba(37,99,235,0.15)",
                        line=dict(width=0),
                        name="Intervalo nominal 95% de reseñas"
                    ))

                fig.add_trace(go.Scatter(
                    x=test["fecha"],
                    y=test["real"],
                    mode="lines+markers",
                    name="Real",
                    line=dict(color=NAVY, width=3)
                ))

                fig.add_trace(go.Scatter(
                    x=test["fecha"],
                    y=test["prediccion"],
                    mode="lines+markers",
                    name="Predicho",
                    line=dict(color=BLUE, dash="dash")
                ))

                st.plotly_chart(
                    style_chart(
                        fig,
                        "Valores observados y pronosticados"
                    ),
                    width="stretch",
                    theme=None, key="exec_test_chart"
                )

            with st.expander(
                "Ver separacion temporal y metodologia"
            ):

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
                    hide_index=True,
                    width="stretch"
                )

                st.write(
                    "Validacion elige el modelo; calibracion "
                    "estima la incertidumbre; test mide "
                    "el rendimiento fuera de la seleccion."
                )

                st.write(
                    "WAPE no es una probabilidad de acierto. "
                    "La cobertura historica tampoco garantiza "
                    "cobertura futura."
                )

    # ==============================================
    # 04. TOURISM SCENARIO SIMULATOR
    # ==============================================

    with tabs[3]:

        st.subheader("De resenas a demanda turistica")

        st.latex(
            r"E=\frac{R}{\rho},"
            r"\qquad P=\frac{RG}{\rho}"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Resenas", number(value))

        c2.metric(
            "Estancias",
            number(estimates["estancias_estimadas"])
        )

        c3.metric(
            "Llegadas",
            number(estimates["llegadas_estimadas"])
        )

        rates = np.arange(0.20, 0.91, 0.05)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=100*rates,
            y=value*guests/rates,
            mode="lines+markers",
            name="Llegadas estimadas",
            line=dict(color=BLUE, width=3)
        ))

        fig.add_trace(go.Scatter(
            x=[rho*100],
            y=[estimates["llegadas_estimadas"]],
            mode="markers",
            name="Escenario elegido",
            marker=dict(color=GREEN, size=14)
        ))

        fig = style_chart(
            fig,
            "Sensibilidad a la tasa de resenas",
            "Llegadas estimadas"
        )

        fig.update_xaxes(
            title="Estancias que publican resena (%)"
        )

        st.plotly_chart(
            fig,
            width="stretch",
            theme=None, key="exec_sensitivity"
        )

        st.info(
            "Si una menor proporcion de estancias deja "
            "resenas, se necesitan mas estancias para "
            "explicar el mismo volumen de resenas."
        )

        st.caption(
            "Este grafico representa escenarios matematicos, "
            "no probabilidades de demanda."
        )

    # ==============================================
    # 05. MARKET AND SCIENTIFIC CONTEXT
    # ==============================================

    with tabs[4]:

        st.subheader("Contexto del mercado")

        row = market[
            market["barrio"] == barrio
        ].iloc[0]

        a, b, c = st.columns(3)

        a.metric(
            "Alojamientos registrados",
            number(row["alojamientos"])
        )

        b.metric(
            "Capacidad registrada",
            number(row["capacidad_total"])
        )

        c.metric(
            "Capacidad media",
            number(row["capacidad_media"], 2)
        )

        st.metric(
            "Disponibilidad anunciada",
            percent(
                context["disponibilidad_media_pct"]
            )
        )

        if not context["calendario_completo"]:

            st.warning(
                "El calendario no cubre completamente "
                "el periodo solicitado."
            )

        st.info(
            "La oferta y disponibilidad proceden de una "
            "captura de Inside Airbnb. Un alojamiento no "
            "disponible puede estar reservado o bloqueado."
        )

        st.subheader("Limitaciones cientificas")

        st.write(
            "- Las resenas son una variable proxy.\n"
            "- No observamos reservas confirmadas.\n"
            "- Las llegadas no son personas unicas.\n"
            "- Los parametros turisticos son supuestos.\n"
            "- Los intervalos no garantizan cobertura futura.\n"
            "- No se incorporan variables exogenas.\n"
            "- La fecha de resena puede diferir de la estancia."
        )

        st.markdown(
            "[Repositorio cientifico en GitHub]"
            "(https://github.com/alexroma1126-design/"
            "valencia-airbnb-intelligence-lab)"
        )

    # ==============================================

    # EXECUTIVE_EXTENSIONS_V31

    with tabs[5]:

        render_territory(
            frequency=frequency,
            target=target,
            future_mode=future_mode,
            rho=rho,
            guests=guests
        )

    with tabs[6]:

        render_math(
            barrio=barrio,
            frequency=frequency,
            target=target,
            future_mode=future_mode,
            y=y,
            summary=summary,
            value=value,
            rho=rho,
            guests=guests
        )

        st.divider()

        render_report(
            barrio=barrio,
            period=period_label(target, frequency),
            future_mode=future_mode,
            summary=summary,
            value=value,
            rho=rho,
            guests=guests
        )


    # EXPORT
    # ==============================================

    st.divider()

    export = {
        "barrio": barrio,
        "consulta": mode,
        "frecuencia": frequency,
        "periodo": str(target.date()),
        "resenas": value,
        **(summary or {}),
        **estimates,
        **context
    }

    st.download_button(
        "Exportar resultado de la consulta",
        pd.DataFrame([export]).to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name=(
            f"valencia_{frequency}_{target:%Y%m%d}.csv"
        ),
        mime="text/csv",
        key="exec_export"
    )

    st.caption(
        "Valencia Intelligence Studio | "
        "Motor predictivo v3.0.0 | "
        "Mat. Héctor Román"
    )


if __name__ == "__main__":

    main()
