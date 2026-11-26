import pandas as pd
import streamlit as st
import plotly.express as px

from src.analytics import (
    descriptive_statistics,
    temporal_reviews,
    moving_average,
    temporal_decomposition,
    autocorrelation_analysis,
    available_stay
)


def render_advanced_tabs(
    tabs, listings, calendar, reviews,
    selected, cal, rev
):

    # 06. LABORATORIO MATEMATICO

    with tabs[5]:

        st.header("Laboratorio matemático")

        st.subheader("Estadística descriptiva de precios")

        st.dataframe(
            descriptive_statistics(
                selected["price_numeric"]
            ).round(3).to_frame("Valor"),
            width="stretch"
        )

        st.divider()

        st.subheader("Análisis de series temporales")

        col1, col2 = st.columns(2)

        with col1:
            frequency = st.selectbox(
                "Frecuencia temporal",
                ["Mensual", "Semanal", "Diaria"],
                key="advanced_frequency"
            )

        with col2:
            unique = st.radio(
                "Unidad de análisis",
                ["Reseñas únicas", "Asociaciones"],
                key="advanced_count"
            ) == "Reseñas únicas"

        freq = {
            "Mensual": "MS",
            "Semanal": "W-SUN",
            "Diaria": "D"
        }[frequency]

        period = {
            "Mensual": 12,
            "Semanal": 52,
            "Diaria": 7
        }[frequency]

        if rev.empty:

            st.warning("No existen reseñas para esta zona.")

        else:

            series = temporal_reviews(
                rev,
                frequency=freq,
                unique=unique
            )

            # Excluir periodos extremos potencialmente incompletos
            series = series.iloc[1:-1]

            if len(series) < 3:

                st.warning(
                    "No existen suficientes periodos."
                )

            else:

                years = sorted(series.index.year.unique())

                selected_years = st.multiselect(
                    "Años de análisis",
                    years,
                    default=years[-5:],
                    key="advanced_years"
                )

                if selected_years:

                    # Mantener un intervalo temporal continuo
                    first_year = min(selected_years)
                    last_year = max(selected_years)

                    series = series[
                        (series.index.year >= first_year)
                        & (series.index.year <= last_year)
                    ]

                    st.caption(
                        f"Intervalo continuo: "
                        f"{first_year} - {last_year}"
                    )

                    if len(series) < 3:

                        st.warning(
                            "Intervalo demasiado corto."
                        )

                    else:

                        window = st.slider(
                            "Ventana de media móvil",
                            min_value=2,
                            max_value=min(30, len(series)),
                            value=3,
                            key="advanced_window"
                        )

                        temporal = pd.DataFrame({
                            "Observado": series,
                            "Media móvil": moving_average(
                                series, window
                            )
                        })

                        st.latex(
                            r"MA_k(t)=\frac{1}{k}"
                            r"\sum_{j=0}^{k-1}Y_{t-j}"
                        )

                        fig = px.line(
                            temporal,
                            title="Serie temporal y media móvil"
                        )

                        fig.update_layout(
                            hovermode="x unified",
                            xaxis_title="Fecha",
                            yaxis_title="Reseñas"
                        )

                        st.plotly_chart(
                            fig,
                            width="stretch"
                        )

                        st.subheader("Descomposición STL")

                        st.latex(r"Y_t=T_t+S_t+R_t")

                        decomposition = temporal_decomposition(
                            series,
                            period=period
                        )

                        if decomposition is None:

                            st.warning(
                                "Datos insuficientes para STL."
                            )

                        else:

                            fig = px.line(
                                decomposition,
                                facet_row=None,
                                title="Componentes de la serie"
                            )

                            st.plotly_chart(
                                fig,
                                width="stretch"
                            )

                            for column in [
                                "tendencia",
                                "estacionalidad",
                                "residuo"
                            ]:

                                fig = px.line(
                                    decomposition,
                                    y=column,
                                    title=column.capitalize()
                                )

                                st.plotly_chart(
                                    fig,
                                    width="stretch"
                                )

                        st.subheader("Autocorrelación")

                        correlations = autocorrelation_analysis(
                            series,
                            lags=36
                        )

                        if correlations is not None:

                            left, right = st.columns(2)

                            with left:

                                st.plotly_chart(
                                    px.bar(
                                        correlations,
                                        x="retardo",
                                        y="ACF",
                                        title="ACF"
                                    ),
                                    width="stretch"
                                )

                            with right:

                                st.plotly_chart(
                                    px.bar(
                                        correlations,
                                        x="retardo",
                                        y="PACF",
                                        title="PACF"
                                    ),
                                    width="stretch"
                                )

                        st.download_button(
                            "Exportar análisis temporal",
                            temporal.to_csv(),
                            file_name="analisis_temporal.csv",
                            mime="text/csv",
                            key="advanced_export"
                        )

        st.info(
            "Las reseñas constituyen un indicador indirecto "
            "de actividad turística, no de reservas reales. "
            "Los ciclos estacionales son hipótesis exploratorias."
        )

    # 07. COMPARADOR DE BARRIOS

    with tabs[6]:

        st.header("Comparador de barrios")

        options = sorted(
            listings["neighbourhood_cleansed"]
            .dropna()
            .unique()
        )

        chosen = st.multiselect(
            "Seleccionar barrios",
            options,
            default=[
                x for x in [
                    "RUSSAFA",
                    "CAMPANAR",
                    "BENIFERRI"
                ]
                if x in options
            ],
            key="compare_neighborhoods"
        )

        rows = []

        for name in chosen:

            zone = listings[
                listings["neighbourhood_cleansed"] == name
            ]

            ids = zone["id"]

            zone_cal = calendar[
                calendar["listing_id"].isin(ids)
            ]

            zone_rev = reviews[
                reviews["listing_id"].isin(ids)
            ]

            rows.append({
                "Barrio": name,
                "Alojamientos": len(zone),
                "Anfitriones": zone["host_id"].nunique(),
                "Precios válidos": zone["price_numeric"].count(),
                "Precio medio": zone["price_numeric"].mean(),
                "Precio mediano": zone["price_numeric"].median(),
                "Desviación estándar": zone["price_numeric"].std(),
                "Disponibilidad (%)": (
                    zone_cal["available_numeric"].mean() * 100
                ),
                "Reseñas únicas": zone_rev["id"].nunique()
            })

        if rows:

            comparison = pd.DataFrame(rows)

            st.dataframe(
                comparison.round(2),
                width="stretch",
                hide_index=True
            )

            indicator = st.selectbox(
                "Indicador a comparar",
                list(comparison.columns[1:]),
                key="comparison_indicator"
            )

            st.plotly_chart(
                px.bar(
                    comparison,
                    x="Barrio",
                    y=indicator,
                    color="Barrio",
                    text_auto=".2f",
                    title=f"Comparación: {indicator}"
                ),
                width="stretch"
            )

            prices = listings[
                listings["neighbourhood_cleansed"]
                .isin(chosen)
            ].dropna(subset=["price_numeric"])

            st.plotly_chart(
                px.box(
                    prices,
                    x="neighbourhood_cleansed",
                    y="price_numeric",
                    color="neighbourhood_cleansed",
                    title="Distribución de precios por barrio"
                ),
                width="stretch"
            )

            if (comparison["Alojamientos"] < 30).any():

                st.warning(
                    "Algunos barrios tienen pocos alojamientos. "
                    "Interpretar los resultados considerando "
                    "el tamaño de cada grupo."
                )

            st.download_button(
                "Exportar comparación",
                comparison.to_csv(index=False),
                file_name="comparacion_barrios.csv",
                mime="text/csv",
                key="comparison_export"
            )

        else:

            st.info("Selecciona al menos un barrio.")

    # 08. ESTANCIAS COMPLETAS

    with tabs[7]:

        st.header("Disponibilidad durante estancias completas")

        st.caption(
            "Consulta histórica del calendario recopilado. "
            "No representa disponibilidad en tiempo real."
        )

        first_date = cal["date"].min().date()
        last_date = cal["date"].max().date()

        col1, col2, col3 = st.columns(3)

        with col1:

            checkin = st.date_input(
                "Fecha de llegada",
                value=first_date,
                min_value=first_date,
                max_value=last_date,
                key="advanced_checkin"
            )

        with col2:

            nights = st.number_input(
                "Número de noches",
                min_value=1,
                max_value=90,
                value=3,
                key="advanced_nights"
            )

        with col3:

            guests = st.number_input(
                "Número de personas",
                min_value=1,
                max_value=15,
                value=2,
                key="advanced_guests"
            )

        budget = st.number_input(
            "Presupuesto de referencia por noche",
            min_value=0,
            max_value=50000,
            value=500,
            step=25,
            key="advanced_budget"
        )

        checkout = (
            pd.Timestamp(checkin)
            + pd.Timedelta(days=int(nights))
        )

        st.write(
            f"**Fecha de salida:** {checkout.date()}"
        )

        if (
            checkout - pd.Timedelta(days=1)
            > pd.Timestamp(last_date)
        ):

            st.error(
                "La estancia supera las fechas del calendario."
            )

        else:

            compatible = available_stay(
                cal,
                checkin,
                int(nights)
            )

            result = selected[
                selected["id"].isin(compatible)
                & (selected["accommodates"] >= guests)
                & (selected["price_numeric"] <= budget)
            ].copy()

            result["coste_referencia"] = (
                result["price_numeric"] * nights
            )

            a, b, c = st.columns(3)

            a.metric(
                "Disponibilidad consecutiva",
                len(compatible)
            )

            b.metric(
                "Cumplen todos los filtros",
                len(result)
            )

            c.metric(
                "Presupuesto total de referencia",
                f"{budget*nights:,.2f}"
            )

            st.latex(
                r"\prod_{t=t_0}^{t_0+k-1}a_{it}=1"
            )

            output = result[[
                "name",
                "room_type",
                "accommodates",
                "price_numeric",
                "coste_referencia",
                "listing_url"
            ]].sort_values("coste_referencia")

            st.dataframe(
                output,
                width="stretch",
                hide_index=True,
                column_config={
                    "listing_url":
                        st.column_config.LinkColumn(
                            "Abrir anuncio"
                        )
                }
            )

            st.download_button(
                "Exportar resultados",
                output.to_csv(index=False),
                file_name="estancias.csv",
                mime="text/csv",
                key="advanced_stay_export"
            )

            st.warning(
                "Los costes son referenciales. "
                "Calendar no contiene precios diarios. "
                "Las restricciones de estancia se verifican "
                "para la fecha de llegada."
            )
