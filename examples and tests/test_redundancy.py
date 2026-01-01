"""
Test to verify anti-redundancy fix
"""
import requests
import json

url = "http://localhost:8000/stream"
data = {
    "message": "Qual o preço da Tesla?",
    "conversation_id": "test_redundancy"
}

print("Testing anti-redundancy fix...")
print(f"Query: {data['message']}\n")

try:
    response = requests.post(url, json=data, stream=True, timeout=60)
    
    if response.status_code == 200:
        print("✅ Stream started\n")
        tool_calls_count = 0
        
        for line in response.iter_lines():
            if line:
                try:
                    event = json.loads(line.decode('utf-8').replace('data: ', ''))
                    if event.get('type') == 'token':
                        print(event['content'], end='', flush=True)
                    elif event.get('type') == 'metadata':
                        metadata = event['metadata']
                        context = metadata.get('context', {})
                        tool_calls = context.get('tool_calls', [])
                        tool_calls_count = len(tool_calls)
                        
                        print(f"\n\n📊 Total tool calls: {tool_calls_count}")
                        
                        # Show unique tools called
                        unique_tools = {}
                        for call in tool_calls:
                            tool_name = call.get('tool_name')
                            args = str(call.get('arguments'))
                            key = f"{tool_name}({args})"
                            unique_tools[key] = unique_tools.get(key, 0) + 1
                        
                        print("\n🔧 Tool calls breakdown:")
                        for tool, count in unique_tools.items():
                            status = "✅ OK" if count == 1 else f"⚠️ REDUNDANT ({count}x)"
                            print(f"   {tool}: {status}")
                        
                        if any(count > 1 for count in unique_tools.values()):
                            print("\n❌ FAILED: Redundant tool calls detected!")
                        else:
                            print("\n✅ SUCCESS: No redundant tool calls!")
                            
                except:
                    pass
        
    else:
        print(f"❌ Error: {response.status_code}")
        
except Exception as e:
    print(f"❌ Exception: {e}")
