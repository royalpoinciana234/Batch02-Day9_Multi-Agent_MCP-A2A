"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Recursive splitter là lựa chọn an toàn cho cả file legal dài và news markdown lẫn lộn.
# Chunk 500 ký tự đủ nhỏ để retrieval chính xác hơn, overlap 50 giữ ngữ cảnh giữa hai đoạn.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Số lượng chiều EMBEDDING_DIM = 384 là cố định và bắt buộc của model all-MiniLM-L6-v2. Kích thước vector 384 chiều này rất thích hợp cho các đoạn văn bản ngắn (dưới 1000 ký tự):
# Đủ lớn để lưu trữ đầy đủ ý nghĩa ngữ nghĩa (semantic meaning) của một đoạn văn ngắn 500 ký tự.
# Đủ nhỏ để việc tính toán độ tương đồng (Cosine Similarity) hoặc lưu trữ trong Vector DB diễn ra cực kỳ nhanh và tốn rất ít RAM/ổ cứng.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Có thể bật Weaviate Cloud bằng VECTOR_STORE=weaviate và set WEAVIATE_URL/WEAVIATE_API_KEY.
VECTOR_STORE = "weaviate"  # "weaviate" | "chromadb" | "faiss" | "local_json"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
    return documents
    # raise NotImplementedError("Implement load_documents")


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i}
            })
    return chunks
    # raise NotImplementedError("Implement chunk_documents")


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    try:
        from sentence_transformers import SentenceTransformer
        # Load the sentence transformer model
        model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Extract text from all chunks
        texts = [chunk["content"] for chunk in chunks]
        
        # Generate embeddings in batch
        embeddings = model.encode(texts)
        
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb.tolist()
            
    except Exception as e:
        print(f"Error encoding with SentenceTransformer: {e}. Falling back to dummy embeddings.")
        # Fallback to simulated embedding
        for chunk in chunks:
            text = chunk["content"][:EMBEDDING_DIM]
            base = [float((ord(char) % 256) / 255.0) for char in text]
            if len(base) < EMBEDDING_DIM:
                base.extend([0.0] * (EMBEDDING_DIM - len(base)))
            chunk["embedding"] = base[:EMBEDDING_DIM]
            
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    import json
    
    if VECTOR_STORE == "local_json":
        vector_store_path = Path(__file__).parent.parent / "data" / "vector_store.json"
        vector_store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(vector_store_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved {len(chunks)} chunks to {vector_store_path}")
        
    elif VECTOR_STORE == "weaviate":
        import weaviate
        from weaviate.classes.config import Configure, Property, DataType
        
        weaviate_url = os.getenv("WEAVIATE_URL")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        
        if weaviate_url and weaviate_api_key:
            print(f"Connecting to Weaviate Cloud: {weaviate_url}")
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_url,
                auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key),
                skip_init_checks=True
            )
        else:
            print("Connecting to Weaviate Local...")
            client = weaviate.connect_to_local()
            
        try:
            # Check if collection already exists
            if client.collections.exists("DrugLawDocs"):
                client.collections.delete("DrugLawDocs")
                
            # Tạo collection
            collection = client.collections.create(
                name="DrugLawDocs",
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="doc_type", data_type=DataType.TEXT),
                ]
            )
            
            # Insert chunks
            with collection.batch.dynamic() as batch:
                for chunk in chunks:
                    batch.add_object(
                        properties={
                            "content": chunk["content"],
                            "source": chunk["metadata"].get("source", ""),
                            "doc_type": chunk["metadata"].get("type", "")
                        },
                        vector=chunk["embedding"]
                    )
            print(f"✓ Successfully indexed {len(chunks)} chunks into Weaviate")
        finally:
            client.close()
    else:
        raise ValueError(f"Unsupported VECTOR_STORE: {VECTOR_STORE}")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
