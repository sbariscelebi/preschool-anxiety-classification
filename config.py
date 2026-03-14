"""
config.py
Global constants and validation note for the pipeline.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGETS    = ["GAD", "SAD"]
WEIGHT_COL = "Sampling Weight"

VALIDATION_NOTE = (
    "LIMITATION: The publicly available PTRTS test dataset (n=307) "
    "contains only demographic variables and does not include raw PAPA items. "
    "Performance metrics were therefore computed on an internal stratified "
    "75/25 split of the PAS training sample (n=917). "
    "External validation on the independent PTRTS cohort was not feasible "
    "with the available public data (Carpenter et al., 2016, "
    "doi:10.1371/journal.pone.0165524)."
)

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 16,
    "axes.titlesize"   : 18,
    "axes.labelsize"   : 20,
    "axes.titleweight" : "bold",
    "xtick.labelsize"  : 20,
    "ytick.labelsize"  : 20,
    "legend.fontsize"  : 14,
    "figure.dpi"       : 300,
    "savefig.dpi"      : 300,
    "pdf.fonttype"     : 42,
    "ps.fonttype"      : 42,
})
