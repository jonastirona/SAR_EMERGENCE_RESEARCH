import json
import numpy as np
from functions import load_all_ar_data

TRAIN_ARS = [
    11130,
    11149,
    11158,
    11162,
    11199,
    11327,
    11344,
    11387,
    11393,
    11416,
    11422,
    11455,
    11619,
    11640,
    11660,
    11678,
    11682,
    11765,
    11768,
    11776,
    11916,
    11928,
    12036,
    12051,
    12085,
    12089,
    12144,
    12175,
    12203,
    12257,
    12331,
    12494,
    12659,
    12778,
    12864,
    12877,
    12900,
    12929,
    13004,
    13085,
    13098,
]

SIZE = 9
RID_OF_TOP = 0  # Must match grid_search.py

print("Loading all 41 training ARs...")
all_power_maps, all_flux, all_cont_int = load_all_ar_data(TRAIN_ARS, SIZE, RID_OF_TOP)

m_scale = (float(np.min(all_power_maps)), float(np.max(all_power_maps)))
flux_scale = (float(np.min(all_flux)), float(np.max(all_flux)))
cont_int_scale = (float(np.min(all_cont_int)), float(np.max(all_cont_int)))

scales = {
    "m_scale": list(m_scale),
    "flux_scale": list(flux_scale),
    "cont_int_scale": list(cont_int_scale),
    "rid_of_top": RID_OF_TOP,
    "num_in": 110,
    "num_pred": 12,
}

out_path = "scales.json"
with open(out_path, "w") as f:
    json.dump(scales, f, indent=2)

print(f"Saved to {out_path}")
print(f"  m_scale:        {m_scale}")
print(f"  flux_scale:     {flux_scale}")
print(f"  cont_int_scale: {cont_int_scale}")
