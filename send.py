import requests

API_URL = "http://127.0.0.1:8000/chat"

payload = {
    "messages": [
        {
            "role": "user",
            "content": "[肥宅議會] grassy_bean: 欸我昨天終於鼓起勇氣約那個學妹去看電影了\n[肥宅議會] caster97: 真假！有料欸，然後勒？\n[肥宅議會] grassy_bean: 她回我說這週末剛好要陪家人...下次一定\n[肥宅議會] phantom.3310: 幫QQ，至少她有回下次一定啦"
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