from pathlib import Path
import sys
import pandas as pd

from streamlit.testing.v1 import AppTest

ROOT = Path.cwd()

sys.path.insert(0, str(ROOT))

from src.forecasting import load_series

app = AppTest.from_file(
    str(ROOT / "app" / "executive_app.py"),
    default_timeout=180
)


def verify():

    assert not app.exception, [
        error.message for error in app.exception
    ]


def metrics():

    return {
        m.label: m.value
        for m in app.metric
    }


print("=" * 65)
print("VALIDACION DE VALENCIA INTELLIGENCE STUDIO")
print("=" * 65)

app.run()
verify()

assert len(app.tabs) == 7

print("[OK] Interfaz empresarial funcional")

app.selectbox(
    key="exec_barrio_mensual"
).select("CABANYAL-CANYAMELAR").run()

verify()

assert metrics()["Reseñas previstas"] == "638"
assert metrics()["Llegadas estimadas"] == "2.552"

print("[OK] Pronostico mensual coincide con version 3.0")

app.slider(
    key="exec_rho"
).set_value(60).run()

verify()

assert metrics()["Llegadas estimadas"] == "2.127"

print("[OK] Simulador responde correctamente")

app.radio(
    key="exec_mode"
).set_value("Historico").run()

verify()

expected = load_series(
    "CABANYAL-CANYAMELAR",
    "mensual"
).loc[pd.Timestamp("2026-03-01")]

formatted = f"{expected:,.0f}".replace(",", ".")

assert metrics()["Reseñas observadas"] == formatted

print("[OK] Consulta historica de marzo 2026")

app.radio(
    key="exec_mode"
).set_value("Pronostico").run()

verify()

app.selectbox(
    key="exec_period_Pronostico_mensual_CABANYAL-CANYAMELAR"
).set_value("2027-03-01").run()

verify()

assert "Reseñas previstas" in metrics()

print("[OK] Consulta predictiva de marzo 2027")

app.radio(
    key="exec_frequency"
).set_value("semanal").run()

verify()

print("[OK] Pronostico semanal")

app.checkbox(
    key="exec_compare"
).check().run()

verify()

print("[OK] Comparacion interactiva de barrios")

print("\nINDICADORES MOSTRADOS")

for label, value in metrics().items():
    print(f"{label}: {value}")

print("\nVALENCIA INTELLIGENCE STUDIO VALIDADO")
