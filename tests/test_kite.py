import json
from pathlib import Path
from typing import List

import pytest
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

from kite.kite import run_indexing, run_query


@pytest.fixture
def sample_json(tmp_path: Path):
    d = tmp_path / "kite"
    d.mkdir()
    data = [
        {"company": "A", "bullets": ["foo bar"]},
        {"company": "B", "bullets": ["baz qux"]},
    ]
    file = d / "data.json"
    file.write_text(json.dumps(data))
    return str(file)


class Test_Bullet_Points:
    document_store = InMemoryDocumentStore()

    def test_indexing_creates_two_documents(self, sample_json: str):
        run_indexing(sample_json, document_store=self.document_store)
        assert self.document_store.count_documents() == 2

    def test_query_returns_matching_bullet(self):
        results: List[Document] = run_query("foo", document_store=self.document_store)
        assert any("foo bar" in doc.content for doc in results)
