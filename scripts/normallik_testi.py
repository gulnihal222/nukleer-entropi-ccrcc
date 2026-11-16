"""Shapiro-Wilk normallik testi: dört birincil entropi ve iki klasik ölçü, iki kohortta.
Kullanım: python normallik_testi.py Data_S1_case_level_measurements.xlsx"""
import sys
import pandas as pd
from scipy.stats import shapiro

d = pd.read_excel(sys.argv[1], sheet_name="case_level")
cols = ["area_um2_entropy", "perimeter_um_entropy", "circularity_entropy", "eccentricity_entropy",
        "area_um2_std", "area_um2_median"]
rows = []
for coh, g in d.groupby("cohort", sort=False):
    for c in cols:
        w, p = shapiro(g[c].dropna())
        rows.append({"cohort": coh, "variable": c, "n": g[c].notna().sum(), "W": round(w, 3), "P": p})
out = pd.DataFrame(rows)
pd.set_option("display.float_format", lambda v: f"{v:.2g}")
print(out.to_string(index=False))
