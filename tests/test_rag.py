"""Unit and integration tests for RAG pipeline, chunking, and retrieval."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_core import (
    DEFAULT_KB_DIR,
    FALLBACK_ANSWER,
    DocumentChunk,
    extract_markdown_sections,
    get_chroma_collection,
    get_embedding_model,
    index_knowledge_base,
    retrieve_relevant_chunks,
    split_chunks,
)


def test_knowledge_base_loading():
    files = list(DEFAULT_KB_DIR.glob("*.md"))
    assert len(files) >= 3, f"Expected at least 3 markdown files in {DEFAULT_KB_DIR}, found {len(files)}"

    for f in files:
        sections = extract_markdown_sections(f)
        assert len(sections) > 0, f"No sections extracted from {f.name}"
        assert all(s.source == f.name for s in sections)


def test_chunking_and_overlap():
    long_text = "Word " * 500  # ~2500 chars
    raw_chunk = DocumentChunk(text=long_text, source="test.md", section="Test")
    split_res = split_chunks([raw_chunk], chunk_size=500, chunk_overlap=100)
    assert len(split_res) > 1
    for chunk in split_res:
        assert len(chunk.text) <= 500
        assert chunk.source == "test.md"


def test_knowledge_base_indexing():
    model = get_embedding_model()
    indexed_count = index_knowledge_base(
        kb_dir=DEFAULT_KB_DIR,
        embedding_model=model,
        force_reindex=True,
    )
    assert indexed_count > 0, "No chunks indexed"

    collection = get_chroma_collection()
    assert collection.count() == indexed_count


def test_retrieval_grounding():
    model = get_embedding_model()
    # Test retrieval for course topics
    results = retrieve_relevant_chunks("What topics are covered in CS 401 modules?", model, top_k=3)
    assert len(results) > 0
    sources = [r.source for r in results]
    assert "course_overview.md" in sources

    # Test retrieval for late submission policy
    results = retrieve_relevant_chunks("What is the penalty for late assignments?", model, top_k=3)
    assert len(results) > 0
    sources = [r.source for r in results]
    assert "academic_policies.md" in sources


def test_fallback_constant():
    assert FALLBACK_ANSWER == "I couldn't find that information in the available documents."


if __name__ == "__main__":
    test_knowledge_base_loading()
    test_chunking_and_overlap()
    test_knowledge_base_indexing()
    test_retrieval_grounding()
    test_fallback_constant()
    print("All RAG tests passed successfully!")
