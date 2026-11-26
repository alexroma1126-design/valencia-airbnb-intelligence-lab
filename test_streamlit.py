from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from streamlit.testing.v1 import AppTest

print("=" * 60)
print("PRUEBAS AUTOMATIZADAS DE STREAMLIT")
print("=" * 60)

app = AppTest.from_file(
    str(ROOT / "app" / "app.py"),
    default_timeout=180
)

app.run()

if len(app.exception):

    print("\nEXCEPCIONES DETECTADAS:")

    for error in app.exception:
        print(error.message)

    raise SystemExit(1)

print("\n[PASS] Ejecucion inicial")

expected_tabs = [
    "01. Exploración",
    "02. Barrios y mapas",
    "03. Precios",
    "04. Series temporales",
    "05. Consultas rápidas",
    "06. Laboratorio matemático",
    "07. Comparador de barrios",
    "08. Estancias completas"
]

actual_tabs = [tab.label for tab in app.tabs]

for name in actual_tabs:
    print("PESTANA:", name)

assert actual_tabs == expected_tabs

print("[PASS] Ocho pestanas integradas")

for barrio in [
    "BENIFERRI",
    "BENICALAP",
    "RUSSAFA",
    "CAMPANAR",
    "ALBORS"
]:

    selector = next(
        element
        for element in app.sidebar.selectbox
        if element.label == "Seleccionar barrio"
    )

    selector.set_value(barrio).run()

    if len(app.exception):

        for error in app.exception:
            print(error.message)

        raise RuntimeError(
            f"Error al seleccionar {barrio}"
        )

    print(f"[PASS] Barrio: {barrio}")

print("\nTODAS LAS PRUEBAS DE STREAMLIT SUPERADAS")
