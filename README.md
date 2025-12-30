# Fred.AI - RAG Agent with ReAct

RAG (Retrieval-Augmented Generation) system specialized in finance and economics, featuring a ReAct agent for iterative reasoning and action.

## 🎯 Features

- **RAG Agent V2**: Semantic search in financial documents
- **ReAct Agent**: Reasoning and action loop with up to 3 iterations
- **Financial Tools**: Stock prices, comparison, document search
- **Intelligent Validation**: Verifies if responses are relevant to the domain
- **Response Synthesis**: Combines multiple iterations without redundancy

## 🏗️ Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Context Agent  │ ← Extracts intent
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│      ReAct Loop (max 3x)        │
│  ┌──────────────────────────┐   │
│  │ 1. Tool Calling Agent    │   │
│  │ 2. Execute 1 Tool        │   │
│  │ 3. ReAct Analysis        │   │
│  │ 4. Decide: Continue/Retry│   │
│  └──────────────────────────┘   │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Response Synth  │ ← Combines responses
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Validation Agent │ ← Validates domain
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Final Response │
└─────────────────┘
```

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/fredstrey/react_agent.git
cd Fred.AI
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create `.env` file:
```env
OPENROUTER_API_KEY=your_key_here
```

### 5. Start Qdrant (Docker)
```bash
docker run -p 6333:6333 qdrant/qdrant
```

## 📦 Project Structure

```
Fred.AI/
├── agents/
│   ├── context_agent.py      # Intent extraction
│   ├── rag_agent_v2.py        # Main RAG Agent
│   ├── react_agent.py         # ReAct: Reasoning + Acting
│   └── validation_agent.py    # Domain validation
├── api/
│   └── api.py                 # FastAPI endpoints
├── core/
│   ├── tool_calling_agent.py  # Base for tool calling
│   ├── execution_context.py   # Execution context
│   ├── registry.py            # Tool registry
│   └── executor.py            # Tool executor
├── embedding_manager/
│   └── embedding_manager.py   # Embeddings manager
├── providers/
│   ├── openrouter.py          # OpenRouter provider
│   └── openrouter_function_caller.py
├── tools/
│   └── rag_tools.py           # RAG tools
└── examples/
    ├── add_finance_docs.py    # Add documents
    └── test_react_agent.py    # ReAct tests
```

## 🛠️ Available Tools

### 1. `search_documents`
Semantic search in financial documents
```python
search_documents(query="What is the Selic rate?")
```

### 2. `get_stock_price`
Get price of ONE stock
```python
get_stock_price(ticker="AAPL")
```

### 3. `compare_stocks`
Compare MULTIPLE stocks
```python
compare_stocks(tickers=["AAPL", "MSFT", "GOOGL"])
```

### 4. `redirect`
Indicates that question is out of scope

## 🎮 Usage

### Start API
```bash
python api/api.py
```

### Make request
```bash
curl -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the price of AAPL and who defines the Selic rate?"}'
```

### Add documents
```bash
python examples/add_finance_docs.py
```

## 🧠 ReAct Agent

The ReAct Agent implements a reasoning and action loop:

### Possible Decisions
- **CONTINUE**: Sufficient information
- **RETRY_WITH_REFINEMENT**: Refine query and try again
- **CALL_DIFFERENT_TOOL**: Call different tool
- **INSUFFICIENT_DATA**: Insufficient data after 3 iterations

### Execution Example
```
Query: "Price of AAPL and who defines Selic?"

Iteration 1: get_stock_price("AAPL") → $273.76
ReAct: Missing answer about Selic → CALL_DIFFERENT_TOOL

Iteration 2: search_documents("Who defines Selic?") → COPOM
ReAct: Both parts answered → CONTINUE

Response: "AAPL: $273.76. COPOM defines the Selic rate."
```

## ⚙️ Configuration

### LLM Models
Configured in `agents/rag_agent_v2.py`:
```python
RAGAgentV2(
    tool_caller_model="xiaomi/mimo-v2-flash:free",
    response_model="xiaomi/mimo-v2-flash:free",
    context_model="xiaomi/mimo-v2-flash:free",
    max_iterations=3  # ReAct iterations
)
```

### Qdrant
```python
EmbeddingManager(
    embedding_model="qwen3-embedding:0.6b",
    qdrant_url="http://localhost:6333",
    collection_name="rag_api"
)
```

## 📊 Implemented Features

✅ ReAct loop with 3 iterations  
✅ Sequential tool execution (semaphore)  
✅ Multi-part query detection  
✅ Automatic query refinement  
✅ Context accumulation between iterations  
✅ Intelligent response synthesis  
✅ Domain validation (finance/economics)  

## 🐛 Troubleshooting

### Qdrant won't connect
```bash
# Check if container is running
docker ps

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

### Invalid API Key
Check `.env` file and configure `OPENROUTER_API_KEY`

### Empty responses
Run `python examples/add_finance_docs.py` to add documents
