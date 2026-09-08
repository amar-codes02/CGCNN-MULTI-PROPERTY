import os
import sys
import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
STRUCTURES_DIR = os.path.join(PROJECT_ROOT, "structures")
os.makedirs(STRUCTURES_DIR, exist_ok=True)

sys.path.insert(0, APP_DIR)
import app as app_module

top5_formulas = ["WB2", "MoC", "Co3O4", "Ti3O5", "Mo2C"]

print("Generating CIF files for Top 5 Host Materials...")
for formula in top5_formulas:
    struct = app_module.get_structure_for_formula(formula)
    cif_filename = f"{formula}.cif"
    cif_filepath = os.path.join(STRUCTURES_DIR, cif_filename)
    cif_content = struct.to(fmt="cif")
    with open(cif_filepath, "w", encoding="utf-8") as f:
        f.write(cif_content)
    print(f" Saved: {cif_filepath} ({len(struct)} atoms)")

    # Also generate adsorbed CIF files for Li2S8, Li2S6, Li2S4, Li2S2, Li2S
    species_list = ["Li2S8", "Li2S6", "Li2S4", "Li2S2", "Li2S"]
    for species in species_list:
        ads_cif_filename = f"{formula}_{species}.cif"
        ads_cif_filepath = os.path.join(STRUCTURES_DIR, ads_cif_filename)
        ads_cif_text = app_module.get_host_adsorbed_cif(struct, species)
        with open(ads_cif_filepath, "w", encoding="utf-8") as f:
            f.write(ads_cif_text)
        print(f"  -> Saved Adsorbed CIF: {ads_cif_filepath}")

print("All Top 5 CIF files successfully generated!")
