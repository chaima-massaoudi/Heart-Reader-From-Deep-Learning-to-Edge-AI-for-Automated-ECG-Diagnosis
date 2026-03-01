"""Quick test: upload all 5 test CSVs and verify predictions."""
import requests

files = ['sample_norm.csv', 'sample_mi.csv', 'sample_sttc.csv', 'sample_cd.csv', 'sample_hyp.csv']
for fname in files:
    path = f'test_files/{fname}'
    r = requests.post('http://localhost:8000/api/predict', files={'file': (fname, open(path, 'rb'), 'text/csv')})
    if r.status_code == 200:
        pred = r.json()['predictions']
        above = {k: v['probability'] for k, v in pred.items() if v['predicted']}
        all_probs = '  '.join(f"{k}={v['probability']:.3f}" for k, v in pred.items())
        print(f"{fname:22s}  predicted: {str(list(above.keys())):30s}  ({all_probs})")
    else:
        print(f"{fname:22s}  ERROR: {r.status_code} {r.text[:100]}")
