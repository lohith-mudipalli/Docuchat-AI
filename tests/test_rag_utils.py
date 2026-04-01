from src.rag_utils import chunk_pages, retrieve_relevant_chunks, get_unique_sources


def test_get_unique_sources_removes_duplicates():
    retrieved_chunks = [
        {"source": "handbook.pdf", "page": 12, "chunk_id": 1, "content": "A"},
        {"source": "handbook.pdf", "page": 12, "chunk_id": 2, "content": "B"},
        {"source": "handbook.pdf", "page": 13, "chunk_id": 3, "content": "C"},
    ]

    result = get_unique_sources(retrieved_chunks)

    assert result == [
        "handbook.pdf - Page 12",
        "handbook.pdf - Page 13",
    ]


def test_chunk_pages_creates_chunks_with_metadata():
    pages_data = [
        {
            "page": 1,
            "source": "sample.pdf",
            "text": "This is a test sentence. " * 200,
        }
    ]

    chunks = chunk_pages(pages_data)

    assert len(chunks) > 1
    assert chunks[0]["page"] == 1
    assert chunks[0]["source"] == "sample.pdf"
    assert "content" in chunks[0]
    assert "chunk_id" in chunks[0]


def test_chunk_pages_skips_blank_text():
    pages_data = [
        {
            "page": 1,
            "source": "sample.pdf",
            "text": "   ",
        }
    ]

    chunks = chunk_pages(pages_data)

    assert chunks == []


class FakeCollection:
    def query(self, query_texts, n_results):
        return {
            "documents": [[
                "Employees may take up to 10 sick leave days.",
                "Unused leave does not carry forward."
            ]],
            "metadatas": [[
                {"page": 12, "source": "handbook.pdf", "chunk_id": 5},
                {"page": 13, "source": "handbook.pdf", "chunk_id": 6},
            ]]
        }


def test_retrieve_relevant_chunks_formats_results_correctly():
    collection = FakeCollection()

    results = retrieve_relevant_chunks(collection, "What is the sick leave policy?", top_k=2)

    assert len(results) == 2
    assert results[0]["page"] == 12
    assert results[0]["source"] == "handbook.pdf"
    assert results[0]["chunk_id"] == 5
    assert "sick leave" in results[0]["content"].lower()

def test_retrieve_relevant_chunks_returns_empty_list_when_no_results():
    class EmptyCollection:
        def query(self, query_texts, n_results):
            return {"documents": [[]], "metadatas": [[]]}

    collection = EmptyCollection()

    results = retrieve_relevant_chunks(collection, "random question", top_k=3)

    assert results == []