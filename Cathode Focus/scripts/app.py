import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
STREAMLIT_APP = os.path.join(PROJECT_ROOT, "streamlit", "app.py")

with open(STREAMLIT_APP, "r", encoding="utf-8") as f:
    code = compile(f.read(), STREAMLIT_APP, "exec")
exec(code, globals())
