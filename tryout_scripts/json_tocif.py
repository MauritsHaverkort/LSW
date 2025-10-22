from pymatgen.core import Structure
import json
with open("tryout_scripts/mp-19770.json", encoding="utf-8") as f:
    d = json.load(f)
Structure.from_dict(d).to(filename="tryout_scripts/hematite.cif")