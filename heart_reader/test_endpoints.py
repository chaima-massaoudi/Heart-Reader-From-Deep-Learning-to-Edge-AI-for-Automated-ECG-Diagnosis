"""Quick end-to-end test of all Heart Reader endpoints."""
import requests

BASE = "http://localhost:8000"
ok = True

print("=" * 55)
print("  Heart Reader — End-to-End Verification")
print("=" * 55)

# 1. Health
print("\n[1] Health Check")
r = requests.get(f"{BASE}/api/health")
h = r.json()
print(f"    Status: {r.status_code}  model_loaded={h['model_loaded']}  samples={h['test_samples_available']}")
ok = ok and r.status_code == 200

# 2. Model Info
print("\n[2] Model Info")
r = requests.get(f"{BASE}/api/model-info")
d = r.json()
perf = d.get("performance", {})
print(f"    Params: {d['num_parameters']:,}")
print(f"    Val AUC:  {perf.get('val_macro_auc', 'N/A')}")
print(f"    Test AUC: {perf.get('test_macro_auc', 'N/A')}")
ok = ok and r.status_code == 200

# 3. Random Sample
print("\n[3] Random Sample")
r = requests.get(f"{BASE}/api/random-sample")
d = r.json()
gt = d["ground_truth"]
preds = {k: round(v["probability"], 3) for k, v in d["predictions"].items()}
print(f"    Ground truth: {gt}")
print(f"    Predictions:  {preds}")
print(f"    Signal shape:  {len(d['signal'])} x {len(d['signal'][0])}")
ok = ok and r.status_code == 200

# 4. Upload: NORM
print("\n[4] Upload sample_norm.csv")
with open("test_files/sample_norm.csv", "rb") as f:
    r = requests.post(f"{BASE}/api/predict", files={"file": ("norm.csv", f, "text/csv")})
d = r.json()
preds = {k: round(v["probability"], 3) for k, v in d["predictions"].items()}
print(f"    Status: {r.status_code}  Predictions: {preds}")
ok = ok and r.status_code == 200

# 5. Upload: MI
print("\n[5] Upload sample_mi.csv")
with open("test_files/sample_mi.csv", "rb") as f:
    r = requests.post(f"{BASE}/api/predict", files={"file": ("mi.csv", f, "text/csv")})
d = r.json()
preds = {k: round(v["probability"], 3) for k, v in d["predictions"].items()}
print(f"    Status: {r.status_code}  Predictions: {preds}")
ok = ok and r.status_code == 200

# 6. Upload: CD
print("\n[6] Upload sample_cd.csv")
with open("test_files/sample_cd.csv", "rb") as f:
    r = requests.post(f"{BASE}/api/predict", files={"file": ("cd.csv", f, "text/csv")})
d = r.json()
preds = {k: round(v["probability"], 3) for k, v in d["predictions"].items()}
print(f"    Status: {r.status_code}  Predictions: {preds}")
ok = ok and r.status_code == 200

# 7. Dashboard
print("\n[7] Dashboard HTML")
r = requests.get(f"{BASE}/")
print(f"    Status: {r.status_code}  Size: {len(r.text):,} chars")
ok = ok and r.status_code == 200

# Summary
print("\n" + "=" * 55)
if ok:
    print("  ALL 7 CHECKS PASSED  --  Project is working well!")
else:
    print("  SOME CHECKS FAILED")
print("=" * 55)
