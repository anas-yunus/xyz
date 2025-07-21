import requests
import json

url = "https://bfhldevapigw.healthrx.co.in/sp-gw/api/openai/v1/embeddings"

headers = {
    "Content-Type": "application/json",
    "x-subscription-key": "sk-spgw-api01-7725518cecd9cb0663448e4489e7693f"
}

payload = json.dumps({
    "model": "text-embedding-3-small",
    "input": ["Test embedding query"]
})

response = requests.post(url, headers=headers, data=payload)

print("Status:", response.status_code)
print("Response:", response.text)
