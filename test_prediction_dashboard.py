from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path.cwd()

print("=" * 70)
print("VALIDACION DEL DASHBOARD PREDICTIVO")
print("=" * 70)

app = AppTest.from_file(
    str(ROOT / "app" / "predictive_app.py"),
    default_timeout=120
)

app.run()

if len(app.exception):

    for error in app.exception:
        print(error.message)

    raise AssertionError(
        "La aplicacion presenta excepciones"
    )

print("[OK] Aplicacion ejecutada sin excepciones")

titles = [
    element.value
    for element in app.title
]

assert any(
    "Inteligencia predictiva" in title
    for title in titles
)

print("[OK] Titulo correcto")

assert len(app.tabs) == 5

print("[OK] Cinco paneles predictivos")

metrics = [
    metric.label
    for metric in app.metric
]

assert "Resenas previstas" in metrics
assert "Estancias estimadas" in metrics
assert "Llegadas estimadas" in metrics

print("[OK] Indicadores principales disponibles")

app.selectbox(
    key="pred_barrio"
).set_value("BENIFERRI").run()

assert not len(app.exception)

print("[OK] Cambio de barrio correcto")

app.radio(
    key="pred_frequency"
).set_value("semanal").run()

assert not len(app.exception)

print("[OK] Cambio de frecuencia correcto")

app.slider(
    key="pred_review_rate"
).set_value(60).run()

assert not len(app.exception)

print("[OK] Cambio de supuestos correcto")

print("\nINDICADORES MOSTRADOS")

for metric in app.metric:

    print(
        f"{metric.label}: {metric.value}"
    )

print("\nDASHBOARD PREDICTIVO VALIDADO")
