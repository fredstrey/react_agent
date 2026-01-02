import ollama
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import uuid


class EmbeddingManager:
    """Embedding manager using Ollama and Qdrant"""
    
    def __init__(
        self,
        embedding_model: str = "qwen3-embedding:0.6b",
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "rag_api",
        filter: Optional[str] = "Constituição da República Federativa do Brasil"
    ):
        """
        Initialize the embedding manager
        
        Args:
            embedding_model: Model for generating embeddings
            qdrant_url: Qdrant server URL
            collection_name: Collection name in Qdrant
            filter: Default source filter for searches
        """
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=qdrant_url)
        
        # Embedding dimension (will be detected automatically)
        self.embedding_dim: Optional[int] = None
        self.filter: Optional[str] = filter
        
    def _get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a text using Ollama
        
        Args:
            text: Text to generate embedding
            
        Returns:
            List of floats representing the embedding
        """
        try:
            response = ollama.embed(
                model=self.embedding_model,
                input=text
            )
            
            # Ollama returns embeddings as list
            embedding = response['embeddings'][0]
            
            # Detect embedding dimension on first run
            if self.embedding_dim is None:
                self.embedding_dim = len(embedding)
                print(f"📊 Embedding dimension detected: {self.embedding_dim}")
            
            return embedding
            
        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")
    
    def initialize_collection(self, recreate: bool = False):
        """
        Initialize the collection in Qdrant
        
        Args:
            recreate: If True, recreate the collection (delete existing data)
        """
        # Generate a test embedding to detect dimension
        if self.embedding_dim is None:
            test_embedding = self._get_embedding("test")
        
        # Check if collection exists
        collections = self.qdrant_client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)
        
        if collection_exists and recreate:
            print(f"🗑️  Deleting existing collection: {self.collection_name}")
            self.qdrant_client.delete_collection(self.collection_name)
            collection_exists = False
        
        if not collection_exists:
            print(f"📦 Creating collection: {self.collection_name}")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
        else:
            print(f"✅ Collection already exists: {self.collection_name}")
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """
        Add documents to the collection
        
        Args:
            documents: List of document texts
            metadatas: Optional list of metadata for each document
        """
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        if len(documents) != len(metadatas):
            raise ValueError("Number of documents and metadatas must be equal")
        
        print(f"📝 Adding {len(documents)} documents...")
        
        points = []
        for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
            # Generate embedding
            embedding = self._get_embedding(doc)
            
            # Create point
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "content": doc,
                    **metadata
                }
            )
            points.append(point)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(documents)} documents...")
        
        # Insert into Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"✅ {len(documents)} documents added successfully!")
    
    def search(self, query: str, top_k: int = 5, filter: Optional[str] = None) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query: Text of the query
            top_k: Number of results to return
            filter: Optional filter by source
            
        Returns:
            List of dictionaries with content, score and metadata
        """
        # Assign default filter from instance if not provided
        if filter is None:
            filter = self.filter

        # DEBUG
        print(f"🔎 [DEBUG] EmbeddingManager.search() called:")
        print(f"   collection_name: {self.collection_name}")
        print(f"   query: '{query}'")
        print(f"   top_k: {top_k}")
        print(f"   filter source: {filter}")
        
        # Generate query embedding
        query_embedding = self._get_embedding(query)
        print(f"   embedding generated: dim={len(query_embedding)}")
        
        # Search in Qdrant using query_points
        from qdrant_client.models import SearchRequest
        
        # Construct filter condition if filter is provided
        query_filter = None
        if filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=filter)
                    )
                ]
            )

        # Search in Qdrant using query_points with query_filter
        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=query_filter 
        ).points
        
        print(f"   results from Qdrant: {len(results)} points")
        
        # DEBUG: Show first result
        if results:
            print(f"   [DEBUG] First result:")
            print(f"      payload keys: {list(results[0].payload.keys())}")
            print(f"      score: {results[0].score}")
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result.payload.get("content", ""),
                "score": result.score,
                "metadata": {k: v for k, v in result.payload.items() if k != "content"}
            })
        
        print(f"   [DEBUG] Formatted results: {len(formatted_results)} itens")
        if formatted_results:
            print(f"   [DEBUG] First item formated has content: {bool(formatted_results[0]['content'])}")
        
        return formatted_results
    
    def get_collection_info(self) -> Dict:
        """
        Returns information about the collection
        
        Returns:
            Dictionary with collection information
        """
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            return {"error": str(e)}
