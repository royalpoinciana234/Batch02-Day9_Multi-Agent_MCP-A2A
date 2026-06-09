"""Stage 1: Direct LLM Calling

The simplest way to use an LLM — send a message, get a response.
No tools, no memory, no agents. Just a direct API call.

This is stateless: the LLM has no access to external data sources,
cannot look things up, and relies entirely on its training data.
"""

import asyncio
import os
import sys

# Allow running directly: python stages/stage_1_direct_llm/main.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from common.llm import get_llm

# QUESTION = "What are the legal consequences if a company breaches a non-disclosure agreement?"
QUESTION = "Nếu một công ty vi phạm thỏa thuận không tiết lộ thông tin, hậu quả pháp lý có thể bao gồm việc bị kiện ra tòa, phải bồi thường thiệt hại cho bên bị vi phạm, và có thể phải chịu các hình phạt khác tùy theo mức độ nghiêm trọng của vi phạm và quy định trong hợp đồng."  # Vietnamese question for testing multilingual capabilities


async def main():
    print("=" * 70)
    print("STAGE 1: Direct LLM Calling")
    print("=" * 70)
    print()
    print("[How it works]")
    print("  1. We send a system prompt + user question directly to the LLM")
    print("  2. The LLM responds from its training data only")
    print("  3. No tools, no retrieval, no external knowledge")
    print()
    print(f"Question: {QUESTION}")
    print("-" * 70)

    llm = get_llm(temperature=0.3)

    messages = [
        SystemMessage(
            content=(
                "You are a legal expert. Provide a clear, concise analysis "
                "of the legal question asked. Keep your response under 300 words."
            )
        ),
        HumanMessage(content=QUESTION),
    ]

    print("\n>>> Calling LLM directly (no tools, no RAG)...\n")
    response = await llm.ainvoke(messages)
    print(response.content)

    print()
    print("-" * 70)
    print("[Limitations of Stage 1]")
    print("  - Stateless: no conversation memory between calls")
    print("  - No tools: cannot search databases or calculate damages")
    print("  - Knowledge cutoff: only knows what was in training data")
    print("  - No grounding: cannot cite specific statutes or current case law")
    print()
    print("Next: Stage 2 adds RAG and tools to ground responses in real data.")
    print("=" * 70)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())