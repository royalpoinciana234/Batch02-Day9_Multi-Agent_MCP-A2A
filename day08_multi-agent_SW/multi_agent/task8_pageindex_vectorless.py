"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pageindex import PageIndexClient, PageIndexAPIError

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ tài liệu PDF trong data/landing/legal/ lên PageIndex.
    """
    if not PAGEINDEX_API_KEY:
        print("⚠ PAGEINDEX_API_KEY không được thiết lập trong .env")
        return

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    
    # Lấy danh sách các tài liệu hiện có trên PageIndex
    try:
        existing_docs = client.list_documents()
        existing_names = {doc["name"] for doc in existing_docs.get("documents", [])}
    except Exception as e:
        print(f"⚠ Lỗi khi lấy danh sách documents từ PageIndex: {e}")
        existing_names = set()

    # Tìm các file PDF trong data/landing/legal/
    legal_dir = Path(__file__).parent.parent / "data" / "landing" / "legal"
    if not legal_dir.exists():
        print(f"⚠ Thư mục {legal_dir} không tồn tại.")
        return

    pdf_files = list(legal_dir.glob("*.pdf"))
    if not pdf_files:
        print("⚠ Không tìm thấy file PDF nào trong data/landing/legal/")
        return

    for pdf_file in pdf_files:
        filename = pdf_file.name
        if filename in existing_names:
            print(f"  ✓ {filename} đã tồn tại trên PageIndex.")
        else:
            print(f"  Uploading: {filename}...")
            try:
                res = client.submit_document(str(pdf_file))
                print(f"  ✓ Uploaded {filename}: {res}")
            except Exception as e:
                print(f"  ❌ Lỗi khi upload {filename}: {e}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not PAGEINDEX_API_KEY:
        print("⚠ PAGEINDEX_API_KEY không được thiết lập trong .env")
        return []

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    
    # 1. Lấy tất cả document IDs
    try:
        docs_res = client.list_documents()
        documents = docs_res.get("documents", [])
    except Exception as e:
        print(f"⚠ Lỗi khi kết nối PageIndex: {e}")
        return []

    if not documents:
        print("⚠ Không có documents nào trên PageIndex để query.")
        return []

    # Lọc ra các document đã completed
    completed_doc_ids = [doc["id"] for doc in documents if doc.get("status") == "completed"]
    if not completed_doc_ids:
        print("⚠ Không có documents nào ở trạng thái 'completed' trên PageIndex.")
        return []

    # 2. Submit query cho từng document
    retrieval_ids = {}
    for doc_id in completed_doc_ids:
        try:
            res = client.submit_query(doc_id, query)
            retrieval_ids[doc_id] = res["retrieval_id"]
        except Exception as e:
            print(f"⚠ Lỗi khi submit query cho doc {doc_id}: {e}")

    if not retrieval_ids:
        return []

    # 3. Poll retrieval status
    def poll_retrieval(doc_id, retrieval_id):
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                ret_res = client.get_retrieval(retrieval_id)
                status = ret_res.get("status")
                if status == "completed":
                    return doc_id, ret_res
                elif status == "failed":
                    print(f"❌ Retrieval {retrieval_id} failed.")
                    return doc_id, None
            except Exception as e:
                print(f"⚠ Lỗi khi check retrieval {retrieval_id}: {e}")
            time.sleep(1.5)
        print(f"⚠ Timeout khi chờ retrieval {retrieval_id} hoàn thành.")
        return doc_id, None

    # Poll in parallel
    results_by_doc = {}
    with ThreadPoolExecutor(max_workers=len(retrieval_ids)) as executor:
        futures = {
            executor.submit(poll_retrieval, doc_id, ret_id): doc_id
            for doc_id, ret_id in retrieval_ids.items()
        }
        for future in as_completed(futures):
            doc_id, ret_res = future.result()
            if ret_res:
                results_by_doc[doc_id] = ret_res

    # 4. Parse nodes và gán score
    merged_results = []
    for doc_id, ret_res in results_by_doc.items():
        nodes = ret_res.get("retrieved_nodes", [])
        for rank, node in enumerate(nodes):
            # Xây dựng content bằng cách join relevant_content
            content_parts = []
            for outer in node.get("relevant_contents", []):
                for item in outer:
                    if isinstance(item, dict) and "relevant_content" in item:
                        content_parts.append(item["relevant_content"])
            
            content = "\n\n".join(content_parts).strip()
            if not content:
                continue

            # Tính score dựa trên rank
            score = 1.0 / (rank + 1)

            # Metadata parsing
            meta_list = node.get("metadata", [])
            metadata = {
                "doc_id": meta_list[0] if len(meta_list) > 0 else doc_id,
                "source": meta_list[1] if len(meta_list) > 1 else "",
                "filename": meta_list[1] if len(meta_list) > 1 else "",
                "description": meta_list[3] if len(meta_list) > 3 else "",
                "title": node.get("title", ""),
                "id": node.get("id", "")
            }

            merged_results.append({
                "content": content,
                "score": score,
                "metadata": metadata,
                "source": "pageindex"
            })

    # 5. Sort theo score descending và trả về top_k
    merged_results.sort(key=lambda x: x["score"], reverse=True)
    return merged_results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")

