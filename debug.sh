import json

path = "token_threshold_reliability_fast/token_threshold_grid_p95_mincount50.json"

with open(path) as f:
    d = json.load(f)

for tid in [198, 6541, 2088, 1099, 1670, 52928, 3524]:
    print(tid, str(tid) in d, d.get(str(tid)))
