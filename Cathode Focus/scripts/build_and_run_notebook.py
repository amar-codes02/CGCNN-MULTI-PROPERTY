import os
import sys

# Add scripts directory to module search path
sys.path.insert(0, os.path.dirname(__file__))

from build_full_notebook import create_notebook

if __name__ == "__main__":
    print(" Building notebook structure from build_full_notebook.py...")
    create_notebook()
    print(" Executing notebook to populate cell outputs and inline figures...")
    os.system(f"{sys.executable} -m jupyter nbconvert --to notebook --execute --inplace notebooks/JARVIS_DFT3D_Data_Extraction.ipynb")
    print(" Notebook execution complete!")
