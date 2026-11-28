from pathlib import Path
from streamlit.testing.v1 import AppTest

app = AppTest.from_file(
    str(Path("app/executive_app.py").resolve()),
    default_timeout=180
)

app.run()

assert not app.exception

app.radio(
    key="exec_frequency"
).set_value("diaria").run()

assert not app.exception, [
    e.message for e in app.exception
]

labels = [m.label for m in app.metric]

assert "Reseñas previstas" in labels
assert "Llegadas estimadas" in labels

print("[OK] Pronostico diario por barrio")

app.selectbox(
    key="exec_barrio_diaria"
).set_value("TODOS").run()

assert not app.exception

assert "Reseñas previstas" in [
    m.label for m in app.metric
]

print("[OK] Pronostico diario de toda Valencia")

app.radio(
    key="exec_mode"
).set_value("Historico").run()

assert not app.exception

assert "Reseñas observadas" in [
    m.label for m in app.metric
]

print("[OK] Consulta historica diaria")

app.radio(
    key="exec_mode"
).set_value("Pronostico").run()

assert not app.exception

app.checkbox(
    key="exec_territory_enabled"
).check().run()

assert not app.exception, [
    e.message for e in app.exception
]

assert "Barrios con resultados" in [
    m.label for m in app.metric
]

print("[OK] Mapa y ranking diario")

app.slider(
    key="exec_rho"
).set_value(60).run()

assert not app.exception

print("[OK] Simulacion turistica diaria")

print("\nINDICADORES DIARIOS")

for metric in app.metric:

    print(metric.label, ":", metric.value)

print("\nDASHBOARD DIARIO VALIDADO")
