import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

nb_path = "notebooks/JARVIS_DFT3D_Data_Extraction.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

try:
    print(f"Executing {nb_path}...")
    ep.preprocess(nb, {'metadata': {'path': 'notebooks'}})
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Notebook executed and saved successfully!")
except Exception as e:
    print(f"Error executing notebook: {e}")
