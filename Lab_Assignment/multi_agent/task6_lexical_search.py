"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import json
import string
from pathlib import Path
from rank_bm25 import BM25Okapi

try:
    from .task4_chunking_indexing import load_documents, chunk_documents
except ImportError:
    from task4_chunking_indexing import load_documents, chunk_documents


def tokenize(text: str) -> list[str]:
    """Làm sạch và tách văn bản thành danh sách từ khóa."""
    text = text.lower()
    # Thay thế các ký tự đặc biệt bằng khoảng trắng để tránh dính từ
    for p in string.punctuation:
        text = text.replace(p, " ")
    return text.split()


def load_corpus() -> list[dict]:
    """Tải corpus từ data/vector_store.json hoặc tự động tạo nếu chưa có."""
    vector_store_path = Path(__file__).parent.parent / "data" / "vector_store.json"
    if vector_store_path.exists():
        try:
            with open(vector_store_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading vector_store.json: {e}. Rebuilding corpus dynamically.")
            
    # Fallback: tự động chunk lại documents nếu không tìm thấy vector_store.json
    try:
        docs = load_documents()
        return chunk_documents(docs)
    except Exception as e:
        print(f"Error building corpus dynamically: {e}")
        return []


# Load corpus
CORPUS: list[dict] = load_corpus()


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


# Khởi tạo BM25 index
if CORPUS:
    bm25 = build_bm25_index(CORPUS)
else:
    bm25 = None


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if not bm25 or not CORPUS:
        print("Warning: BM25 index is not initialized or corpus is empty. Returning empty list.")
        return []

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Lấy danh sách index sắp xếp giảm dần theo điểm số
    scored_indices = sorted(
        [(score, idx) for idx, score in enumerate(scores)],
        key=lambda x: x[0],
        reverse=True
    )

    results = []
    for score, idx in scored_indices:
        if len(results) >= top_k:
            break
        # Chỉ giữ lại kết quả có độ trùng khớp thực sự (score > 0)
        if score > 0.0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(score),
                "metadata": CORPUS[idx]["metadata"]
            })

    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
