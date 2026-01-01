"""
Quick test script for the /stream endpoint
"""
import requests
import json

url = "http://localhost:8000/stream"
data = {
    "message": "O que é a Selic?",
    "conversation_id": "test123"
}

print("Testing /stream endpoint...")
print(f"Query: {data['message']}\n")

try:
    response = requests.post(url, json=data, stream=True, timeout=30)
    
    if response.status_code == 200:
        print("✅ Stream started successfully!\n")
        print("Response:")
        for line in response.iter_lines():
            if line:
                try:
                    event = json.loads(line.decode('utf-8').replace('data: ', ''))
                    if event.get('type') == 'token':
                        print(event['content'], end='', flush=True)
                    elif event.get('type') == 'metadata':
                        print(f"\n\n✅ Metadata: {json.dumps(event['metadata'], indent=2)}")
                    elif event.get('type') == 'error':
                        print(f"\n❌ Error: {event['content']}")
                except:
                    pass
        print("\n\n✅ Stream completed!")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"❌ Exception: {e}")
