"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import os
import json
from pathlib import Path
try:
    from .task4_chunking_indexing import VECTOR_STORE, embed_chunks
except ImportError:
    from task4_chunking_indexing import VECTOR_STORE, embed_chunks


def cosine_similarity(v1, v2) -> float:
    """Tính toán cosine similarity giữa hai vector."""
    try:
        import numpy as np
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))
    except ImportError:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    # Bước 1: Embed query bằng cùng model ở Task 4
    query_chunk = [{"content": query}]
    embed_chunks(query_chunk)
    query_embedding = query_chunk[0]["embedding"]

    results = []

    # Bước 2: Query vector store
    if VECTOR_STORE == "local_json":
        vector_store_path = Path(__file__).parent.parent / "data" / "vector_store.json"
        if not vector_store_path.exists():
            print(f"Warning: Vector store file {vector_store_path} not found. Returning empty list.")
            return []
            
        with open(vector_store_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            
        # Tính toán similarity cho từng chunk
        for chunk in chunks:
            score = cosine_similarity(query_embedding, chunk["embedding"])
            results.append({
                "content": chunk["content"],
                "score": score,
                "metadata": chunk["metadata"]
            })
            
        # Sắp xếp giảm dần theo score
        results.sort(key=lambda x: x["score"], reverse=True)
        
    elif VECTOR_STORE == "weaviate":
        import weaviate
        from weaviate.classes.query import MetadataQuery
        
        weaviate_url = os.getenv("WEAVIATE_URL")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        
        if weaviate_url and weaviate_api_key:
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_url,
                auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key),
                skip_init_checks=True
            )
        else:
            client = weaviate.connect_to_local()
            
        try:
            if client.collections.exists("DrugLawDocs"):
                collection = client.collections.get("DrugLawDocs")
                search_response = collection.query.near_vector(
                    near_vector=query_embedding,
                    limit=top_k,
                    return_metadata=MetadataQuery(distance=True)
                )
                for obj in search_response.objects:
                    # Chuyển đổi distance sang cosine similarity (Weaviate nearVector trả về distance = 1 - cosine_similarity)
                    # Xem tài liệu Weaviate: score = 1 - distance
                    distance = obj.metadata.distance if obj.metadata.distance is not None else 1.0
                    score = 1.0 - distance
                    results.append({
                        "content": obj.properties.get("content", ""),
                        "score": float(score),
                        "metadata": {
                            "source": obj.properties.get("source", ""),
                            "type": obj.properties.get("doc_type", "")
                        }
                    })
            else:
                print("Warning: Collection 'DrugLawDocs' does not exist in Weaviate. Returning empty list.")
        finally:
            client.close()
            
    else:
        raise ValueError(f"Unsupported VECTOR_STORE: {VECTOR_STORE}")

    # Bước 3: Return top_k results
    return results[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
