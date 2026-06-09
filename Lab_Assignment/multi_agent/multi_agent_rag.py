"""
Multi-Agent RAG System.

Biến pipeline RAG đơn (task10_generation.py) thành hệ multi-agent
với Supervisor + 3 Workers bằng Google Antigravity SDK.

Workers:
    1. ReorderWorker  — reorder_for_llm + format_context
    2. FallbackWorker — generate_fallback_answer
    3. CitationWorker — generate_with_citation (LLM call)

Usage:
    python multi_agent_rag.py "Hình phạt cho tội tàng trữ ma tuý?"
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Thêm thư mục chứa script này vào sys.path để import hoạt động ổn định
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

# ─── Import từ task9 & task10 (cùng thư mục) ────────────────────────────────
from task9_retrieval_pipeline import retrieve
from task10_generation import (
    reorder_for_llm,
    format_context,
    generate_fallback_answer,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
)

# ─── Google Antigravity SDK ──────────────────────────────────────────────────
from google.antigravity import Agent, LocalAgentConfig, types
from google.antigravity.types import CustomSystemInstructions
from google.antigravity.hooks import policy


# =============================================================================
# 1. PYDANTIC MODELS — Shared State & Trace
# =============================================================================

class TraceEntry(BaseModel):
    """Một bước trong execution trace."""
    agent: str
    action: str
    input_summary: str
    output_summary: str
    timestamp: str
    duration_ms: float


class RAGSharedState(BaseModel):
    """State chia sẻ giữa Supervisor và các Workers."""
    raw_query: str
    refined_query: Optional[str] = None
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    reordered_context: Optional[str] = None
    final_answer: Optional[str] = None
    generation_mode: str = "none"       # 'citation' | 'fallback'
    sources_used: int = 0
    trace: List[TraceEntry] = Field(default_factory=list)


# =============================================================================
# 2. CUSTOM TOOLS — Mỗi tool wrap 1 hàm từ task10
# =============================================================================

def reorder_and_format(chunks_json: str) -> str:
    """Reorder chunks để tránh lost-in-the-middle rồi format thành context string.

    Áp dụng reorder_for_llm() để đặt chunks quan trọng ở đầu/cuối,
    rồi format_context() để gắn [Document i | Source] labels.

    Args:
        chunks_json: JSON string chứa list chunks từ retrieval.
    """
    chunks = json.loads(chunks_json)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    return context


def fallback_generate(query: str, chunks_json: str) -> str:
    """Tạo câu trả lời rule-based từ chunks khi không có LLM API key.

    Trích xuất thông tin từ top-3 chunks và gán citation tương ứng.

    Args:
        query: Câu hỏi gốc của user.
        chunks_json: JSON string chứa list chunks.
    """
    chunks = json.loads(chunks_json)
    return generate_fallback_answer(query, chunks)


def citation_generate(query: str, formatted_context: str) -> str:
    """Gọi LLM (OpenAI) để generate câu trả lời tiếng Việt có citation.

    Sử dụng formatted context đã được reorder để tránh lost-in-the-middle.

    Args:
        query: Câu hỏi đã refined.
        formatted_context: Context string đã format với source labels.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        user_message = f"Context:\n{formatted_context}\n\n---\n\nQuestion: {query}"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[LLM Error: {e}]"


# =============================================================================
# 3. WORKER CONFIGS
# =============================================================================

def create_reorder_config() -> LocalAgentConfig:
    """Config cho ReorderWorker — có custom tool reorder_and_format."""
    return LocalAgentConfig(
        system_instructions=CustomSystemInstructions(
            text=(
                "Bạn là ReorderWorker. Nhiệm vụ duy nhất: "
                "dùng tool reorder_and_format để sắp xếp lại chunks và format context. "
                "Gọi tool với chunks_json nhận được, rồi trả về nguyên văn kết quả."
            )
        ),
        tools=[reorder_and_format],
        policies=[policy.allow_all()],
    )


def create_fallback_config() -> LocalAgentConfig:
    """Config cho FallbackWorker — có custom tool fallback_generate."""
    return LocalAgentConfig(
        system_instructions=CustomSystemInstructions(
            text=(
                "Bạn là FallbackWorker. Nhiệm vụ duy nhất: "
                "dùng tool fallback_generate để tạo câu trả lời dự phòng. "
                "Gọi tool với query và chunks_json nhận được, rồi trả về nguyên văn kết quả."
            )
        ),
        tools=[fallback_generate],
        policies=[policy.allow_all()],
    )


def create_citation_config() -> LocalAgentConfig:
    """Config cho CitationWorker — có custom tool citation_generate."""
    return LocalAgentConfig(
        system_instructions=CustomSystemInstructions(
            text=(
                "Bạn là CitationWorker. Nhiệm vụ duy nhất: "
                "dùng tool citation_generate để gọi LLM tạo câu trả lời có citation. "
                "Gọi tool với query và formatted_context nhận được, rồi trả về nguyên văn kết quả."
            )
        ),
        tools=[citation_generate],
        policies=[policy.allow_all()],
    )


def create_slang_config() -> LocalAgentConfig:
    """Config cho SlangWorker — kết nối với local slang MCP server."""
    mcp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slang_mcp.py")
    
    mcp_servers = [
        types.McpStdioServer(
            name="slang_server",
            command=sys.executable,
            args=[mcp_path],
        )
    ]
    
    return LocalAgentConfig(
        system_instructions=CustomSystemInstructions(
            text=(
                "Bạn là SlangWorker. Nhiệm vụ duy nhất: "
                "sử dụng công cụ lookup_slang của MCP server để kiểm tra xem "
                "câu truy vấn có chứa từ lóng ma túy nào không. "
                "Gọi tool với query nhận được và trả về nguyên văn bản giải nghĩa của các từ lóng được phát hiện."
            )
        ),
        mcp_servers=mcp_servers,
        policies=[policy.allow_all()],
    )


# =============================================================================
# 4. WORKER RUNNER — Chạy 1 worker, ghi trace
# =============================================================================

async def run_worker(
    config: LocalAgentConfig,
    prompt: str,
    worker_name: str,
    action: str,
    state: RAGSharedState,
) -> str:
    """Chạy 1 worker agent, ghi trace entry, trả kết quả text."""
    start = time.time()

    async with Agent(config) as agent:
        response = await agent.chat(prompt)
        result = await response.text()

    elapsed_ms = (time.time() - start) * 1000

    state.trace.append(TraceEntry(
        agent=worker_name,
        action=action,
        input_summary=prompt[:100] + ("..." if len(prompt) > 100 else ""),
        output_summary=result[:200] + ("..." if len(result) > 200 else ""),
        timestamp=datetime.now().isoformat(),
        duration_ms=round(elapsed_ms, 1),
    ))

    return result


# =============================================================================
# 5. ORCHESTRATE — Supervisor logic
# =============================================================================

async def orchestrate(query: str, top_k: int = 5) -> RAGSharedState:
    """
    Supervisor: điều phối tuần tự 4 workers (bao gồm SlangWorker dùng MCP).

    Flow:
        0. SlangWorker (MCP) → translate slangs
        1. retrieve() → raw chunks
        2. ReorderWorker → formatted context
        3. Route: CitationWorker (nếu có key) hoặc FallbackWorker
        4. Tổng hợp kết quả, trả state + trace
    """
    state = RAGSharedState(raw_query=query)

    # ── trace[0]: Supervisor routing ─────────────────────────────────────
    start_total = time.time()
    state.trace.append(TraceEntry(
        agent="Supervisor",
        action="routing",
        input_summary=query,
        output_summary="Bắt đầu pipeline multi-agent",
        timestamp=datetime.now().isoformat(),
        duration_ms=0,
    ))

    # ── Step 0: SlangWorker (MCP) ─────────────────────────────────────────
    print(f"\n💬 Supervisor → SlangWorker (MCP): Checking query for drug slangs...")
    slang_result = await run_worker(
        config=create_slang_config(),
        prompt=f"Kiểm tra từ lóng trong câu truy vấn sau:\nQuery: {query}",
        worker_name="SlangWorker",
        action="slang_lookup",
        state=state,
    )
    state.refined_query = slang_result
    print(f"   ✅ Slang check complete: {slang_result.strip()}")

    # ── Step 1: Retrieve (gọi trực tiếp, không qua worker) ──────────────
    print(f"\n🔍 Supervisor: Retrieving top-{top_k} chunks...")
    retrieve_start = time.time()
    # Nếu phát hiện từ lóng, dùng kèm giải nghĩa để tối ưu hóa tìm kiếm
    retrieval_query = query
    if "Phát hiện từ lóng" in slang_result:
        retrieval_query = f"{query} ({slang_result})"
        
    chunks = retrieve(retrieval_query, top_k=top_k)
    retrieve_ms = (time.time() - retrieve_start) * 1000

    state.retrieved_chunks = chunks
    state.sources_used = len(chunks)

    state.trace.append(TraceEntry(
        agent="Supervisor",
        action="retrieve",
        input_summary=retrieval_query[:100] + ("..." if len(retrieval_query) > 100 else ""),
        output_summary=f"Retrieved {len(chunks)} chunks in {retrieve_ms:.0f}ms",
        timestamp=datetime.now().isoformat(),
        duration_ms=round(retrieve_ms, 1),
    ))
    print(f"   ✅ Got {len(chunks)} chunks ({retrieve_ms:.0f}ms)")

    if not chunks:
        state.final_answer = "Không tìm thấy tài liệu liên quan."
        state.generation_mode = "none"
        return state

    # ── Step 2: ReorderWorker ────────────────────────────────────────────
    chunks_json = json.dumps(chunks, ensure_ascii=False)
    print(f"\n🔀 Supervisor → ReorderWorker: Reorder {len(chunks)} chunks...")

    context = await run_worker(
        config=create_reorder_config(),
        prompt=f"Reorder và format chunks sau:\n{chunks_json}",
        worker_name="ReorderWorker",
        action="reorder",
        state=state,
    )
    state.reordered_context = context
    print(f"   ✅ Context formatted ({len(context)} chars)")

    # ── Step 3: Route — CitationWorker hoặc FallbackWorker ───────────────
    api_key = os.getenv("OPENAI_API_KEY", "")
    is_valid_key = api_key and api_key != "sk-xxx" and api_key.strip() != ""

    if is_valid_key:
        # → CitationWorker
        print(f"\n📝 Supervisor → CitationWorker: Generating cited answer...")
        state.generation_mode = "citation"

        answer = await run_worker(
            config=create_citation_config(),
            prompt=(
                f"Trả lời câu hỏi sau bằng tiếng Việt có citation:\n"
                f"Query: {query}\n"
                f"Slang Info: {slang_result}\n"
                f"Context: {context}"
            ),
            worker_name="CitationWorker",
            action="citation_generation",
            state=state,
        )
    else:
        # → FallbackWorker
        print(f"\n🛡️ Supervisor → FallbackWorker: Generating fallback answer...")
        state.generation_mode = "fallback"

        answer = await run_worker(
            config=create_fallback_config(),
            prompt=(
                f"Tạo câu trả lời từ chunks:\n"
                f"Query: {query}\n"
                f"Slang Info: {slang_result}\n"
                f"Chunks: {chunks_json}"
            ),
            worker_name="FallbackWorker",
            action="fallback_generation",
            state=state,
        )

    state.final_answer = answer
    print(f"   ✅ Answer generated ({len(answer)} chars)")

    # ── trace[-1]: Supervisor finalize ───────────────────────────────────
    total_ms = (time.time() - start_total) * 1000
    state.trace.append(TraceEntry(
        agent="Supervisor",
        action="finalize",
        input_summary="Collected all worker outputs",
        output_summary=f"Done. mode={state.generation_mode}, sources={state.sources_used}",
        timestamp=datetime.now().isoformat(),
        duration_ms=round(total_ms, 1),
    ))

    return state


# =============================================================================
# 6. PRINT TRACE — In trace đẹp ra console
# =============================================================================

AGENT_ICONS = {
    "Supervisor": "🎯",
    "SlangWorker": "💬",
    "ReorderWorker": "🔀",
    "FallbackWorker": "🛡️",
    "CitationWorker": "📝",
}

def print_trace(state: RAGSharedState):
    """In execution trace và kết quả ra console."""
    print("\n")
    print("═" * 70)
    print("                      EXECUTION TRACE")
    print("═" * 70)

    for i, entry in enumerate(state.trace, 1):
        icon = AGENT_ICONS.get(entry.agent, "❓")
        print(
            f"[{i}] {icon} {entry.agent:<18} │ "
            f"{entry.action:<22} │ {entry.duration_ms:>8.0f}ms"
        )
        print(f"    Input:  {entry.input_summary}")
        print(f"    Output: {entry.output_summary}")
        print()

    print("═" * 70)
    print(f"  Mode: {state.generation_mode} | Sources: {state.sources_used}")
    print("═" * 70)

    print(f"\n{'─' * 70}")
    print("FINAL ANSWER:")
    print(f"{'─' * 70}")
    print(state.final_answer)
    print(f"{'─' * 70}\n")


# =============================================================================
# 7. CLI ENTRY POINT
# =============================================================================

async def main():
    """Entry point: nhận query từ CLI hoặc dùng default."""
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?"

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║              MULTI-AGENT RAG SYSTEM                            ║")
    print("║         Supervisor + 3 Workers (Antigravity SDK)               ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"\n📥 Query: {query}")

    state = await orchestrate(query)
    print_trace(state)

    # Export state as JSON for debugging
    state_json = state.model_dump_json(indent=2)
    output_path = os.path.join(os.path.dirname(__file__), "last_run_state.json")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(state_json)
    print(f"💾 State saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
