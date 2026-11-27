from pathlib import Path
import sys
import numpy as np

ROOT = Path.cwd()

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from streamlit.testing.v1 import AppTest

from executive_extensions import (
    territorial_data,
    geographic_data
)

print("=" * 65)
print("VALIDACION DE EXTENSIONES EMPRESARIALES")
print("=" * 65)

df, excluded = territorial_data(
    "mensual",
    "2026-11-01",
    True
)

assert not df.empty
assert not df["Barrio"].duplicated().any()
assert (df["Resenas"] >= 0).all()
assert "TODOS" not in df["Barrio"].values

print("[OK] Ranking territorial consistente")

cabanyal = df.loc[
    df["Barrio"] == "CABANYAL-CANYAMELAR",
    "Resenas"
].iloc[0]

assert np.isclose(cabanyal, 638)

print("[OK] Resultados territoriales coinciden con v3.0")

geo, mapping = geographic_data()

assert geo is not None
assert len(mapping) > 0

print("[OK] Limites geograficos encontrados")
print("Correspondencias geograficas:", len(mapping))
print("Barrios con pronostico:", len(df))
print("Barrios excluidos:", len(excluded))

app = AppTest.from_file(
    str(ROOT / "app" / "executive_app.py"),
    default_timeout=180
)

app.run()

assert not app.exception, [
    e.message for e in app.exception
]

assert len(app.tabs) == 7

print("[OK] Siete modulos ejecutivos disponibles")

app.checkbox(
    key="exec_territory_enabled"
).check().run()

assert not app.exception, [
    e.message for e in app.exception
]

print("[OK] Mapa y ranking ejecutados")

metrics = {
    m.label: m.value
    for m in app.metric
}

assert metrics["Llegadas estimadas"] == "2.156"

print("[OK] Resultado ejecutivo conservado")

app.radio(
    key="exec_territory_metric"
).set_value("Llegadas").run()

assert not app.exception

print("[OK] Indicador territorial interactivo")

app.selectbox(
    key="exec_math_model_mensual_A"
).select("B_K3").run()

assert not app.exception

metrics = {
    m.label: m.value
    for m in app.metric
}

assert metrics["Llegadas estimadas"] == "2.156"

print("[OK] Laboratorio no modifica el modelo oficial")

print("\nEXTENSIONES EMPRESARIALES VALIDADAS")
