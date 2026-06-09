"""
Day 8 v2 — Multi-Agent RAG Pipeline
Automated Test Suite for the Multi-Agent System.

Run:
    pytest tests/test_multi_agent.py -v
"""

import json
import os
import sys
import unittest
from pathlib import Path

# Project root setup
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# Import multi-agent modules
try:
    from multi_agent.multi_agent_rag import (
        TraceEntry,
        RAGSharedState,
        reorder_and_format,
        fallback_generate,
        citation_generate,
        create_reorder_config,
        create_fallback_config,
        create_citation_config,
        create_slang_config,
    )
except ImportError as e:
    raise ImportError(f"Cannot import multi_agent_rag: {e}. Check if you are running from the correct directory.")

class TestMultiAgentSchema(unittest.TestCase):
    """Verify that the Shared State and Trace schemas are correctly defined."""

    def test_trace_entry_fields(self):
        """TraceEntry pydantic model has correct fields and types."""
        entry = TraceEntry(
            agent="TestAgent",
            action="test_action",
            input_summary="test input",
            output_summary="test output",
            timestamp="2026-06-09T20:00:00",
            duration_ms=123.4,
        )
        self.assertEqual(entry.agent, "TestAgent")
        self.assertEqual(entry.action, "test_action")
        self.assertEqual(entry.input_summary, "test input")
        self.assertEqual(entry.output_summary, "test output")
        self.assertEqual(entry.timestamp, "2026-06-09T20:00:00")
        self.assertEqual(entry.duration_ms, 123.4)

    def test_shared_state_fields(self):
        """RAGSharedState pydantic model matches specified schema."""
        state = RAGSharedState(
            raw_query="Test query",
            refined_query="Refined query",
            retrieved_chunks=[{"content": "chunk 1", "score": 0.9, "metadata": {"source": "doc1"}}],
            reordered_context="Formatted doc1 context",
            final_answer="Answer here",
            generation_mode="citation",
            sources_used=1,
            trace=[],
        )
        self.assertEqual(state.raw_query, "Test query")
        self.assertEqual(state.refined_query, "Refined query")
        self.assertEqual(len(state.retrieved_chunks), 1)
        self.assertEqual(state.reordered_context, "Formatted doc1 context")
        self.assertEqual(state.final_answer, "Answer here")
        self.assertEqual(state.generation_mode, "citation")
        self.assertEqual(state.sources_used, 1)
        self.assertEqual(len(state.trace), 0)


class TestCustomTools(unittest.TestCase):
    """Test individual custom tools in isolation."""

    def test_reorder_and_format_tool(self):
        """reorder_and_format correctly merges, reorders and formats context."""
        chunks = [
            {"content": "Content 1", "score": 0.9, "metadata": {"source": "doc1.md", "type": "legal"}},
            {"content": "Content 2", "score": 0.8, "metadata": {"source": "doc2.md", "type": "legal"}},
            {"content": "Content 3", "score": 0.7, "metadata": {"source": "doc3.md", "type": "news"}},
        ]
        chunks_json = json.dumps(chunks)
        result = reorder_and_format(chunks_json)
        
        # Check that it returns formatted documents
        self.assertIn("[Document 1", result)
        self.assertIn("doc1.md", result)
        self.assertIn("doc2.md", result)
        self.assertIn("doc3.md", result)

    def test_fallback_generate_tool(self):
        """fallback_generate extracts summaries from top chunks when OpenAI is disabled."""
        chunks = [
            {"content": "Important legal data about ma tuy", "score": 0.9, "metadata": {"source": "luat_ma_tuy.md"}},
            {"content": "News report about artist arrest", "score": 0.8, "metadata": {"source": "news_artist.md"}},
        ]
        chunks_json = json.dumps(chunks)
        result = fallback_generate("test query", chunks_json)
        
        self.assertIn("[Fallback Answer Mode", result)
        self.assertIn("luat_ma_tuy.md", result)
        self.assertIn("news_artist.md", result)
        self.assertIn("Important legal data", result)

    def test_citation_generate_error_handling(self):
        """citation_generate handles missing or invalid API keys gracefully without crashing."""
        # Force a bad API key to test resilience
        os.environ["OPENAI_API_KEY"] = "invalid_key_for_testing"
        
        result = citation_generate("Some question", "Some formatted context")
        self.assertTrue(result.startswith("[LLM Error:") or "Error" in result)


class TestWorkerConfigs(unittest.TestCase):
    """Ensure worker agents have proper configs, tools and instructions."""

    def test_reorder_worker_config(self):
        """ReorderWorker config has custom tool and instructions."""
        config = create_reorder_config()
        self.assertIn(reorder_and_format, config.tools)
        self.assertIn("ReorderWorker", config.system_instructions.text)

    def test_fallback_worker_config(self):
        """FallbackWorker config has custom tool and instructions."""
        config = create_fallback_config()
        self.assertIn(fallback_generate, config.tools)
        self.assertIn("FallbackWorker", config.system_instructions.text)

    def test_citation_worker_config(self):
        """CitationWorker config has custom tool and instructions."""
        config = create_citation_config()
        self.assertIn(citation_generate, config.tools)
        self.assertIn("CitationWorker", config.system_instructions.text)

    def test_slang_worker_config(self):
        """SlangWorker config has MCP server settings and instructions."""
        config = create_slang_config()
        self.assertTrue(len(config.mcp_servers) > 0)
        self.assertIn("SlangWorker", config.system_instructions.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
