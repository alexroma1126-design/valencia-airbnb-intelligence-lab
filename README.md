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
