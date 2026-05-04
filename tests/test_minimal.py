import requests
try:
    print("Testing minimal server on 8001...")
    r = requests.get("http://localhost:8001/health", timeout=2)
    print(f"Minimal Server Status: {r.status_code}")
    print(r.json())
except Exception as e:
    print(f"Minimal Server FAILED: {e}")
