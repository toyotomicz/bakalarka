import requests
from pathlib import Path

json_path = Path("test.json")
json_path.write_text('{"test": 1}')

bin_name = "benchmark-results-2026"
file_url = f"https://filebin.net/{bin_name}/{json_path.name}"

print(f"Posting to: {file_url}")

resp = requests.post(
    file_url,
    data=open(json_path, "rb"),
    headers={
        "Content-Type": "application/octet-stream",
        "Accept": "application/json",
    },
    timeout=30,
)

print(f"Status: {resp.status_code}")
print(f"Response: {resp.text}")