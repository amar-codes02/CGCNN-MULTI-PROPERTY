import os
import sys
import json
import nbformat

nb_path = "notebooks/JARVIS_DFT3D_Data_Extraction.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

os.chdir(os.path.dirname(os.path.abspath(nb_path)))

print("Executing all code cells from notebook...")
full_code = []
for idx, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        # Remove display calls or keep them safe
        code = cell.source
        full_code.append(f"# --- CELL {idx} ---\n" + code)

exec_str = "\n\n".join(full_code)
# Replace display(...) with print(...) or safe execution
exec_str = exec_str.replace("display(", "print(")

exec_globals = {"__name__": "__main__"}
exec(exec_str, exec_globals)
print("All notebook code cells executed successfully!")
