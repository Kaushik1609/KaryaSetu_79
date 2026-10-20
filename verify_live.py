import urllib.request
import json

frontend_ok = False
for url in ["http://localhost:5173/", "http://127.0.0.1:5173/", "http://[::1]:5173/"]:
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            print(f"Frontend Status: {resp.status} (Vite React UI is running on {url})")
            frontend_ok = True
            break
    except Exception:
        continue
if not frontend_ok:
    print("Frontend error: Unable to connect on localhost, 127.0.0.1, or [::1]")

try:
    with urllib.request.urlopen("http://localhost:8000/api/model-info") as resp:
        info = json.loads(resp.read().decode("utf-8"))
        print(f"Backend Status: {resp.status} | Model: {info.get('model_type')} ({info.get('model_version')})")
except Exception as e:
    print("Backend error:", e)

try:
    with urllib.request.urlopen("http://localhost:8000/api/historical-events") as resp:
        data = json.loads(resp.read().decode("utf-8"))
        cases = data.get("results", [])
        print(f"Historical Events Count: {len(cases)}")
        for c in cases[:5]:
            print(f"  * {c.get('event_id')} -> {c.get('name')} | P(Bust): {c.get('model_bust_probability', 0)*100:.1f}% | Predicted Bust: {c.get('model_predicted_bust')}")
except Exception as e:
    print("Backend events error:", e)

try:
    with urllib.request.urlopen("http://localhost:8000/api/model-performance") as resp:
        m = json.loads(resp.read().decode("utf-8"))
        print(f"Metrics -> ROC-AUC: {m.get('roc_auc')} | Calibrated Brier Score: {m.get('brier_calibrated')}")
except Exception as e:
    print("Metrics error:", e)
