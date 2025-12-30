"""
RAG Tools using the framework pattern with decorators
"""
from typing import Dict, Any
from pydantic import BaseModel, Field
import yfinance as yf

from core.decorators import tool
from embedding_manager.embedding_manager import EmbeddingManager


# =========================
# SCHEMAS
# =========================

class SearchArgs(BaseModel):
    """Search arguments"""
    query: str = Field(..., description="Query de busca")
    top_k: int = Field(default=3, description="Number of results")


class StockPriceArgs(BaseModel):
    """Stock price arguments"""
    ticker: str = Field(..., description="Stock ticker (e.g: AAPL, GOOGL, MSFT)")
    period: str = Field(default="1mo", description="Historical period (1d, 5d, 1mo, 3mo, 6mo, 1y)")


class CompareStocksArgs(BaseModel):
    """Stock comparison arguments"""
    tickers: list = Field(..., description="List of stock tickers (e.g: ['AAPL', 'GOOGL', 'MSFT'])")
    period: str = Field(default="1mo", description="Comparison period (1d, 5d, 1mo, 3mo, 6mo, 1y)")


# =========================
# TOOLS
# =========================

# Variável global para armazenar o embedding_manager
_embedding_manager = None


def initialize_rag_tools(embedding_manager: EmbeddingManager):
    """
    Initialize RAG tools with the embedding manager
    
    Args:
        embedding_manager: EmbeddingManager instance
    """
    global _embedding_manager
    _embedding_manager = embedding_manager


@tool(
    name="search_documents",
    description="Search relevant documents in the knowledge base using semantic search"
)
def search_documents(query: str) -> Dict[str, Any]:
    """
    Search relevant documents in the knowledge base
    
    Args:
        query: User query
        top_k: Number of results to return
        
    Returns:
        Dictionary with search results
    """
    if _embedding_manager is None:
        return {
            "success": False,
            "error": "EmbeddingManager not initialized. Call initialize_rag_tools() first."
        }
    
    
    
    try:
        # DEBUG: Log da query recebida
        print(f"🔍 [DEBUG] search_documents called with:")
        print(f"   query: '{query}' (len={len(query)})")

        # Search in Qdrant
        results = _embedding_manager.search(query=query, top_k=3)
        
        print(f"   [DEBUG] EmbeddingManager returned: {len(results)} results")
        
        # Format response
        chunks = [
            {
                "content": r["content"],
                "score": r["score"],
                "metadata": r["metadata"]
            }
            for r in results
        ]
        
        # Debug: Print chunks
        if chunks:
            print(f"\n📄 [DEBUG] Chunks found:")
            for i, chunk in enumerate(chunks, 1):
                content_preview = chunk["content"][:15000] + "..." if len(chunk["content"]) > 15000 else chunk["content"]
                print(f"   Chunk {i}:")
                print(f"      Score: {chunk['score']:.4f}")
                print(f"      Content: {content_preview}")
                print(f"      Metadata: {chunk['metadata']}")
        
        return {
            "success": True,
            "query": query,
            "results": chunks,
            "total_found": len(chunks)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query
        }


@tool(
    name="get_stock_price",
    description="Obtém informações de preço de UMA ÚNICA ação. Use APENAS quando a pergunta menciona UMA ação específica. Para comparar múltiplas ações, use compare_stocks."
)
def get_stock_price(ticker: str, period: str = "1mo") -> Dict[str, Any]:
    """
    Get price information for stocks
    
    Args:
        ticker: Stock ticker (e.g: AAPL, GOOGL, MSFT)
        period: Historical period (1d, 5d, 1mo, 3mo, 6mo, 1y)
        
    Returns:
        Dictionary with stock information
    """
    print(f"🔍 [DEBUG] get_stock_price called with:")
    print(f"   ticker: {ticker} (type: {type(ticker)})")
    print(f"   period: {period}")
    try:
        ticker = ticker.upper()
        
        # Get stock data using yfinance
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            return {
                "success": False,
                "error": f"Não foi possível encontrar dados para a ação {ticker}",
                "ticker": ticker,
                "message": f"Não foi possível encontrar dados de preço para a empresa {ticker}. Verifique se o ticker está correto ou se a ação ainda está listada em bolsa."
            }
        
        info = stock.info
        last_close = float(hist['Close'].iloc[-1])
        
        # Calculate change
        if len(hist) > 1:
            first_close = float(hist['Close'].iloc[0])
            change = last_close - first_close
            change_percent = (change / first_close) * 100
        else:
            change = 0
            change_percent = 0
        
        result = {
            "success": True,
            "ticker": ticker,
            "company_name": info.get('longName', ticker),
            "current_price": round(last_close, 2),
            "currency": info.get('currency', 'USD'),
            "period": period,
            "change": round(change, 2),
            "change_percent": round(change_percent, 2),
            "period_high": round(float(hist['High'].max()), 2),
            "period_low": round(float(hist['Low'].min()), 2),
            "market_cap": info.get('marketCap'),
            "sector": info.get('sector'),
            "summary": f"{info.get('longName', ticker)} está cotado a ${round(last_close, 2)} {info.get('currency', 'USD')}. No período de {period}, a ação variou {round(change_percent, 2)}%."
        }
        print(f"   ✅ [DEBUG] Resposta: {result}")
        return result
    except Exception as e:
        error_msg = str(e)
        # Friendly error message for common errors
        if "delisted" in error_msg.lower() or "no price data" in error_msg.lower():
            user_message = f"Não foi possível encontrar dados de preço para {ticker}. A ação pode ter sido removida da bolsa (delisted) ou o ticker pode estar incorreto."
        else:
            user_message = f"Erro ao buscar informações de {ticker}. Verifique se o ticker está correto."
        
        return {
            "success": False,
            "error": error_msg,
            "ticker": ticker,
            "message": user_message
        }


@tool(
    name="compare_stocks",
    description="Compara MÚLTIPLAS ações (2 ou mais). Use quando a pergunta menciona VÁRIAS ações, palavras como 'compare', 'melhor', 'pior', 'ranking', ou lista múltiplos tickers."
)
def compare_stocks(tickers: list, period: str = "1mo") -> Dict[str, Any]:
    """
    Compare the performance of multiple stocks
    
    Args:
        tickers: List of stock symbols (ex: ['AAPL', 'GOOGL', 'MSFT'])
        period: Period for comparison (1d, 5d, 1mo, 3mo, 6mo, 1y)
        
    Returns:
        Comparison of stock performance
    """
    print(f"🔍 [DEBUG] compare_stocks chamada com:")
    print(f"   tickers: {tickers} (type: {type(tickers)})")
    print(f"   period: {period}")
    try:
        # Handle both list and string formats
        if isinstance(tickers, list):
            ticker_list = [str(t).strip().upper() for t in tickers]
        elif isinstance(tickers, str):
            ticker_list = [t.strip().upper() for t in tickers.split(',')]
        else:
            return {
                "success": False,
                "error": f"Formato inválido para tickers: {type(tickers)}"
            }
        
        print(f"   [DEBUG] ticker_list processed: {ticker_list}")
        
        if len(ticker_list) < 2:
            return {
                "success": False,
                "error": "Forneça pelo menos 2 tickers para comparar"
            }
        
        results = []
        failed_tickers = []
        
        for ticker in ticker_list:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                
                if not hist.empty:
                    first_close = float(hist['Close'].iloc[0])
                    last_close = float(hist['Close'].iloc[-1])
                    change_percent = ((last_close - first_close) / first_close) * 100
                    
                    results.append({
                        "ticker": ticker,
                        "start_price": round(first_close, 2),
                        "end_price": round(last_close, 2),
                        "change_percent": round(change_percent, 2)
                    })
                else:
                    failed_tickers.append(ticker)
            except Exception as e:
                failed_tickers.append(ticker)
        
        # If no stocks have data
        if not results:
            return {
                "success": False,
                "error": "Não encontrei dados para nenhuma das ações",
                "message": f"Não consegui encontrar dados de preço para as ações: {', '.join(ticker_list)}. Verifique se os tickers estão corretos."
            }
        
        # If some stocks failed, mention in the response
        results.sort(key=lambda x: x['change_percent'], reverse=True)
        
        best = results[0]
        worst = results[-1]
        
        summary = f"No período de {period}, {best['ticker']} teve o melhor desempenho com {best['change_percent']}%, enquanto {worst['ticker']} teve o pior com {worst['change_percent']}%."
        
        if failed_tickers:
            summary += f" Nota: Não foi possível obter dados para: {', '.join(failed_tickers)}."
        
        result = {
            "success": True,
            "period": period,
            "stocks": results,
            "best_performer": best,
            "worst_performer": worst,
            "failed_tickers": failed_tickers,
            "summary": summary
        }
        print(f"   ✅ [DEBUG] Response: {result}")
        return result
    except Exception as e:
        error_msg = str(e)
        if "delisted" in error_msg.lower() or "no price data" in error_msg.lower():
            user_message = "Não consegui encontrar dados de preço para as ações solicitadas. Algumas podem ter sido removidas da bolsa (delisted) ou os tickers podem estar incorretos."
        else:
            user_message = f"Erro ao comparar ações. Verifique se os tickers estão corretos."
        
        return {
            "success": False,
            "error": error_msg,
            "message": user_message
        }

@tool(
    name="redirect",
    description="Use quando a pergunta NÃO tem relação com finanças, economia, mercado financeiro, ações ou investimentos. Indica que o assunto está fora do escopo."
)
def redirect(reason: str = "fora do escopo") -> Dict[str, Any]:
    """
    Redirect questions outside the scope of finance/economy
    
    Args:
        reason: Reason for redirection
        
    Returns:
        Dictionary indicating redirection
    """
    return {
        "success": True,
        "redirected": True,
        "reason": reason,
        "message": "Pergunta fora do escopo de finanças e economia"
    }