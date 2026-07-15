import itertools
import json
from pathlib import Path

d = json.loads((Path(__file__).resolve().parent / "witness_n24.json").read_text())
n = d["n"]
col = d["coloring"]
def color(a,b):
    return col[f"{min(a,b)}-{max(a,b)}"]
redK4=0
for q in itertools.combinations(range(n),4):
    if all(color(a,b)==1 for a,b in itertools.combinations(q,2)): redK4+=1
blueK5=0
for q in itertools.combinations(range(n),5):
    if all(color(a,b)==0 for a,b in itertools.combinations(q,2)): blueK5+=1
print(f"n={n} seed={d['seed']}: red K4={redK4}  blue K5={blueK5}  ->",
      "VALID R(4,5)>=25 WITNESS" if redK4==0 and blueK5==0 else "INVALID")
