import os
import sys

# Set up project root in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# Execute main Streamlit app script dynamically on every rerun
script_path = os.path.join(SCRIPTS_DIR, "app.py")
with open(script_path, "r", encoding="utf-8") as f:
    code = compile(f.read(), script_path, "exec")

exec(code, globals())
