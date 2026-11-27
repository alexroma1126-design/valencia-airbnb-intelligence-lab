# Valencia Airbnb Intelligence Lab

## Información académica

- Autor: Mat. Héctor Román
- Programa: Maestría en Ciencia de Datos
- Asignatura: PB - Aprendizaje de Máquina
- Docente: Luis Fernando Zhinin Vera
- Proyecto: 1
- Versión: 2.0

## Descripción

Laboratorio computacional para la exploración, análisis
estadístico, análisis geoespacial y estudio de series
temporales de alojamientos turísticos en Valencia, España.

El sistema permite analizar dinámicamente los 84 barrios
representados en la base de alojamientos.

## Fuente de datos

Inside Airbnb:
https://insideairbnb.com/get-the-data/

## Dimensiones de los datos

| Dataset | Registros | Variables originales |
|---|---:|---:|
| Listings | 7968 | 90 |
| Calendar | 2908320 | 5 |
| Reviews | 509513 | 6 |

- Total de registros: 3425801
- Barrios representados: 84
- Anfitriones únicos: 3215
- Identificadores únicos de reseñas: 458154
- Elementos geográficos GeoJSON: 88

## Funcionalidades

1. Exploración y calidad de datos.
2. Análisis geoespacial de alojamientos.
3. Estadística descriptiva de precios.
4. Series temporales.
5. Consultas rápidas.
6. Laboratorio matemático.
7. Comparador de barrios.
8. Disponibilidad durante estancias completas.

## Fundamentos matemáticos

- Estadística descriptiva.
- Distribuciones y percentiles.
- Varianza y desviación estándar.
- Agregación temporal.
- Medias móviles.
- Descomposición STL.
- Autocorrelación ACF.
- Autocorrelación parcial PACF.
- Matrices binarias de disponibilidad.
- Operaciones relacionales.

## Validación computacional

- 56 pruebas de auditoría superadas.
- 0 pruebas fallidas.
- 8 módulos de Streamlit integrados.
- Pruebas automatizadas sobre cinco barrios.
- Verificación SHA-256 de las bases principales.
- Error máximo de reconstrucción STL: 9.095e-13.

## Estructura del proyecto

    app/
        app.py
        advanced_tabs.py
        geographic.py

    src/
        analytics.py
        download_data.py
        inspect_data.py
        query_engine.py
        audit_reviews.py

    data/
        raw/
        processed/

    results/
        figures/
        tables/

    test_analytics.py
    validate_project.py
    test_streamlit.py
    requirements.txt
    requirements-lock.txt

## Instalación en Windows

    python -m venv .venv

    .\.venv\Scripts\Activate.ps1

    python -m pip install -r requirements-lock.txt

## Descarga de datos

    python src/download_data.py

## Validación

    python test_analytics.py
    python validate_project.py
    python test_streamlit.py

## Ejecución

    python -m streamlit run app/app.py --server.address 127.0.0.1

Aplicación local:

http://127.0.0.1:8501

## Limitaciones metodológicas

- La disponibilidad no equivale necesariamente a ocupación.
- Las reseñas no equivalen al número real de huéspedes.
- Un identificador de reseña puede asociarse a varios anuncios.
- Los precios registrados no equivalen necesariamente a
  precios efectivos de contratación.
- Calendar no contiene precios diarios.
- Las coordenadas geográficas pueden ser aproximadas.
- Los resultados históricos corresponden a los anuncios
  representados en la recopilación de datos.
- Las pruebas automatizadas no sustituyen completamente
  la validación visual en el navegador.

## Trabajo futuro

La etapa predictiva se desarrollará después de definir
el barrio de estudio, la variable objetivo y la consigna
académica del docente.

## Estado

Versión 2.0: laboratorio exploratorio y analítico validado.
Modelado predictivo pendiente de definición.

## Informe cientifico-tecnico

El proyecto incluye un informe academico sobre el analisis
estadistico, geoespacial y temporal de Valencia.

- Autor: Mat. Hector Roman
- Version documental: 2.1.0
- Base analitica: v2.0.0
- Figuras cientificas: 10

[Consultar el informe PDF](informes/Valencia_Airbnb_Intelligence_Lab_Informe.pdf)

### Reproducir el informe

Generar las figuras:

    python generate_report_figures.py

Compilar el documento:

    cd informes
    pdflatex main.tex
    pdflatex main.tex
    pdflatex main.tex

La compilacion requiere una distribucion LaTeX compatible,
por ejemplo MiKTeX o TeX Live.

La version 2.1.0 incorpora la documentacion cientifica
sin modificar los resultados analiticos de la version 2.0.0.


<!-- VALENCIA_V3_START -->

---

# Valencia Airbnb Intelligence Lab v3.0.0

## Predictive Intelligence

**Autor:** Mat. Hector Alexander Roman Potosi

**Programa:** Maestria en Ciencia de Datos

**Modulo:** Aprendizaje de Maquina

## Datos analizados

- 7968 alojamientos.
- 84 barrios.
- 509513 registros de resenas.
- 458154 identificadores unicos de resenas.
- 2908320 registros del calendario.
- Fuente: Inside Airbnb.
- Corte conservador: 11 de septiembre de 2026.

## Modelos predictivos

- Modelo A: naive estacional.
- Modelo B: estacional con crecimiento.
- Seleccion automatica mediante MAE.
- Frecuencias semanal y mensual.
- Validacion cronologica.
- Calibracion independiente.
- Backtesting con origen movil.
- Intervalos nominales del 95%.

## Resultados experimentales

- Pronosticos generados: 144.
- Intervalos disponibles: 144.

## Pronosticos mensuales de noviembre de 2026

| Barrio | Modelo | Resenas | Intervalo 95% | WAPE test |
|---|---|---:|---:|---:|
| BENIFERRI | A | 13 | 3 - 23 | 36.36% |
| CABANYAL-CANYAMELAR | A | 638 | 353 - 923 | 10.11% |
| RUSSAFA | A | 539 | 304 - 774 | 17.70% |
| TODOS | A | 7547 | 5468 - 9626 | 23.63% |


## Plataforma interactiva

1. Exploracion.
2. Barrios y mapas.
3. Precios.
4. Series temporales.
5. Consultas rapidas.
6. Laboratorio matematico.
7. Comparador de barrios.
8. Estancias completas.
9. Prediccion de demanda.

### Paneles predictivos

- Respuesta ejecutiva.
- Pronostico.
- Evaluacion.
- Mercado.
- Metodologia.

## Ejecucion

### Dashboard predictivo independiente

    python -m streamlit run app/predictive_app.py

### Plataforma completa

    python -m streamlit run app/app.py

## Reproducibilidad

Con los datos originales disponibles en data/raw:

    python -m src.prepare_forecast_data
    python -m src.forecasting
    python -m src.backtesting
    python -m src.uncertainty
    python -m src.tourism

## Pruebas automatizadas

    python test_forecast_data.py
    python test_forecasting.py
    python test_backtesting.py
    python test_uncertainty.py
    python test_tourism.py
    python test_prediction_dashboard.py
    python test_integrated_dashboard.py

## Limitaciones cientificas

- Las resenas no equivalen a reservas confirmadas.
- Las llegadas estimadas no representan personas unicas.
- Los parametros turisticos son supuestos modificables.
- La disponibilidad anunciada no demuestra ocupacion.
- El nivel nominal del 95% no garantiza cobertura futura.
- Algunos barrios presentan historia insuficiente.
- Existe un posible sesgo de supervivencia.
- No se incorporan eventos externos ni meteorologia.
- Las estimaciones dependen de la estabilidad temporal.
- La evaluacion preliminar no sustituye la evaluacion definitiva.

## Tecnologias

Python, Pandas, NumPy, PyArrow, Statsmodels, Plotly,
Scikit-learn, Streamlit, Git y LaTeX.

<!-- VALENCIA_V3_END -->


<!-- VALENCIA_V31_START -->

---

# Version 3.1.0 - Valencia Intelligence Studio

Interfaz ejecutiva de inteligencia turistica y
prediccion de demanda.

## Funcionalidades empresariales

- Consultas historicas y predictivas.
- Pronosticos semanales y mensuales.
- Comparacion dinamica de barrios.
- Seleccion interpretable de modelos.
- Simulador de escenarios turisticos.
- Mapa interactivo de demanda.
- Ranking territorial.
- Laboratorio matematico interactivo.
- Resumen ejecutivo automatico.
- Exportacion de resultados CSV y TXT.

## Ejecucion

    python -m streamlit run app/executive_app.py

En Windows, con el entorno virtual preparado:

    .\iniciar_studio.ps1

## Validacion

    python test_executive_dashboard.py
    python test_executive_extensions.py

La plataforma conserva el motor cientifico v3.0.0.

Las estimaciones de huespedes dependen de supuestos
y no representan reservas ni ocupacion observadas.

Los intervalos nominales del 95% no garantizan
cobertura futura.

<!-- VALENCIA_V31_END -->
