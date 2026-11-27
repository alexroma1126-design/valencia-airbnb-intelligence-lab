from pathlib import Path
from streamlit.testing.v1 import AppTest

path = Path.cwd() / "app" / "app.py"

print("=" * 65)
print("VALIDACION DE LA PLATAFORMA INTEGRADA")
print("=" * 65)

app = AppTest.from_file(
    str(path),
    default_timeout=180
)

app.run()

if app.exception:

    for error in app.exception:
        print(error.message)

    raise RuntimeError(
        "Error en la plataforma integrada"
    )

labels = [
    tab.label
    for tab in app.tabs
]

print("\nPESTANAS DISPONIBLES")

for label in labels:
    print("[OK]", label)

assert any(
    "09. Prediccion de demanda" in label
    for label in labels
)

print("\nPLATAFORMA INTEGRADA VALIDADA")
