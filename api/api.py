"""
API FastAPI with ReAct Agent
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse
import json
import uuid
import sys
import os
from typing import AsyncGenerator
from dotenv import load_dotenv
from api_utils import _sync_to_async_generator

# Load environment variables
load_dotenv()

# Adiciona pasta raiz ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api_schemas import ChatRequest, ChatResponse, ProcessPDFRequest, ProcessPDFResponse
from embedding_manager.embedding_manager import EmbeddingManager
from agents.finance_ai import FinanceAI
from agents.legal_ai import LegalAI
from pdf_pipeline.pdf_processor import PDFProcessor
import agents.finance_ai_tools as finance_ai_tools

# Initialize FastAPI
app = FastAPI(
    title="ReAct Agent API",
    description="API with HFSM Agent",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global components
print("🚀 Initializing components...")

# Embedding manager
embedding_manager = EmbeddingManager(
    embedding_model="qwen3-embedding:0.6b",
    qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    collection_name="rag_api",
    filter="Constituição da República Federativa do Brasil"
)

# Initialize collection if necessary
try:
    embedding_manager.initialize_collection(recreate=False)
    print("✅ Qdrant collection initialized")
except Exception as e:
    print(f"⚠️  Warning initializing collection: {e}")

# PDF Processor
pdf_processor = PDFProcessor(embedding_manager)
print("\n✅ PDF Processor initialized!")

# Initialize Finance AI tools with embedding manager
finance_ai_tools.initialize_finance_ai_tools(embedding_manager)
print("✅ Finance AI tools initialized!")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Agent API",
        "version": "2.0.0",
        "models": {
            "tool_caller": "gemma3:1b",
            "embeddings": "qwen3-embedding:0.6b",
            "response_generator": "gemma3:1b"
        },
        "endpoints": {
            "/stream": "POST - Chat with streaming",
            "/health": "GET - Health check",
            "/process_pdf": "POST - Process PDF"
        }
    }


@app.get("/health")
async def health():
    """Health check"""
    try:
        # Check collection
        collection_info = embedding_manager.get_collection_info()
        
        return {
            "status": "healthy",
            "components": {
                "qdrant": "error" not in collection_info,
                "embedding_manager": True
            },
            "collection": {
                "name": collection_info.get("name", embedding_manager.collection_name),
                "documents": collection_info.get("points_count", 0)
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/stream")
async def stream_chat(request: ChatRequest):
    """
    Chat endpoint with async streaming using HFSM Agent.
    
    Now fully async for better performance and concurrency!
    """

    conversation_id = request.conversation_id or str(uuid.uuid4())

    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            # -----------------------------
            # 1. Process chat history
            # -----------------------------
            chat_history = []
            if request.chat_history:
                history_dicts = [msg.model_dump() for msg in request.chat_history]
                chat_history = history_dicts[-6:]  # last 3 turns

            # -----------------------------
            # 2. Initialize Finance.AI agent
            # -----------------------------
            finance_ai = FinanceAI(
                llm_provider="openrouter",
                model="google/gemini-2.0-flash-exp:free"
            )

            # -----------------------------
            # 3. Stream tokens (fully async!)
            # -----------------------------
            async for token in finance_ai.stream(
                query=request.message,
                chat_history=chat_history
            ):
                yield json.dumps({
                    "type": "token",
                    "content": token
                })

            # -----------------------------
            # 4. Final metadata event
            # -----------------------------
            # Collect metadata from agent's last execution context
            if hasattr(finance_ai.agent, 'last_context'):
                ctx = finance_ai.agent.last_context
                
                # Get all metadata (already populated by agent)
                token_usage = await ctx.get_memory("total_usage", {})
                total_requests = await ctx.get_memory("total_requests", 0)
                sources_used = await ctx.get_memory("sources_used", [])
                confidence = await ctx.get_memory("confidence", "-")
            else:
                token_usage = {}
                total_requests = 0
                sources_used = []
                confidence = "-"
            
            yield json.dumps({
                "type": "metadata",
                "content": "",
                "metadata": {
                    "conversation_id": conversation_id,
                    "sources_used": sources_used,
                    "confidence": confidence,
                    "usage": token_usage,
                    "total_requests": total_requests
                }
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield json.dumps({
                "type": "error",
                "content": str(e),
                "metadata": {}
            })

    return EventSourceResponse(generate_stream())

@app.post("/stream_legalai")
async def stream_chat_legalai(request: ChatRequest):
    """
    Chat endpoint with async streaming using HFSM Agent.
    
    Now fully async for better performance and concurrency!
    """

    conversation_id = request.conversation_id or str(uuid.uuid4())

    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            # -----------------------------
            # 1. Process chat history
            # -----------------------------
            chat_history = []
            if request.chat_history:
                history_dicts = [msg.model_dump() for msg in request.chat_history]
                chat_history = history_dicts[-6:]  # last 3 turns

            # -----------------------------
            # 2. Initialize Legal.AI agent
            # -----------------------------
            legal_ai = LegalAI(
                llm_provider="openrouter",
                model="xiaomi/mimo-v2-flash:free",
                embedding_manager=embedding_manager
            )

            # -----------------------------
            # 3. Stream tokens (fully async!)
            # -----------------------------
            async for token in legal_ai.stream(
                query=request.message,
                chat_history=chat_history
            ):
                yield json.dumps({
                    "type": "token",
                    "content": token
                })

            # -----------------------------
            # 4. Final metadata event
            # -----------------------------
            # Collect metadata from agent's last execution context
            if hasattr(legal_ai.agent, 'last_context'):
                ctx = legal_ai.agent.last_context
                
                # Get all metadata (already populated by agent)
                token_usage = await ctx.get_memory("total_usage", {})
                total_requests = await ctx.get_memory("total_requests", 0)
                sources_used = await ctx.get_memory("sources_used", [])
                confidence = await ctx.get_memory("confidence", "-")
            else:
                token_usage = {}
                total_requests = 0
                sources_used = []
                confidence = "-"
            
            yield json.dumps({
                "type": "metadata",
                "content": "",
                "metadata": {
                    "conversation_id": conversation_id,
                    "sources_used": sources_used,
                    "confidence": confidence,
                    "usage": token_usage,
                    "total_requests": total_requests
                }
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield json.dumps({
                "type": "error",
                "content": str(e),
                "metadata": {}
            })

    return EventSourceResponse(generate_stream())

@app.post("/process_pdf", response_model=ProcessPDFResponse)
async def process_pdf(request: ProcessPDFRequest):
    """
    Process a PDF and add to knowledge base
    
    Args:
        request: ProcessPDFRequest with PDF path and parameters
        
    Returns:
        ProcessPDFResponse with processing statistics
    """
    try:
        # Check if file exists
        if not os.path.exists(request.pdf_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.pdf_path}")
        
        # Processa PDF
        stats = pdf_processor.process_pdf(
            pdf_path=request.pdf_path,
            max_tokens=request.max_tokens
        )
        
        return ProcessPDFResponse(
            status="success",
            pdf_file=stats["pdf_file"],
            total_chunks=stats["total_chunks"],
            total_tokens=stats["total_tokens"],
            avg_tokens_per_chunk=stats["avg_tokens_per_chunk"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 70)
    print("🚀 Starting RAG Agent API...")
    print("=" * 70)
    print("\n📍 Endpoints disponíveis:")
    print("   • http://localhost:8000/")
    print("   • http://localhost:8000/stream (POST)")
    print("   • http://localhost:8000/process_pdf (POST)")
    print("   • http://localhost:8000/health")
    print("\n" + "=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
