import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from agents.legal_ai import LegalAI
    print("Successfully imported LegalAI")
    
    agent = LegalAI()
    print("Successfully initialized LegalAI")
    print(f"Persona start: {agent.persona[:50]}...")
    print(f"Tools count: {len(agent.agent.tools)}")
    print(f"Tool name: {agent.agent.tools[0].name}")
    
    # Check Strategy Swap
    strategy_name = agent.agent.engine.contract_strategy.__class__.__name__
    print(f"Contract Strategy: {strategy_name}")
    if strategy_name == "LegalEpistemicContractStrategy":
        print("✅ Strategy Swap SUCCESS")
    else:
        print(f"❌ Strategy Swap FAILED (Got {strategy_name})")
    
    
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
