from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prediction_tab import (
    render_prediction_tab
)

st.set_page_config(
    page_title="Valencia | Inteligencia predictiva",
    page_icon="📊",
    layout="wide"
)

render_prediction_tab()
