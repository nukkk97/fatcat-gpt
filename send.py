import requests

API_URL = "http://127.0.0.1:8000/chat"

payload = {
    "messages": [
        {
            "role": "user",
            "content": ""
        }
    ],
    "temperature": 0.3 
}

print("發送請求中...")
response = requests.post(API_URL, json=payload)

if response.status_code == 200:
    data = response.json()
    print("FatCat:", data["response"])
else:
    print("發生錯誤:", response.text)