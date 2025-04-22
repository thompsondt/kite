import json
import pytest
from haystack.document_stores.in_memory import InMemoryDocumentStore
from kite.kite import run_indexing

@pytest.fixture
def sample_json(tmp_path):
    d = tmp_path / "kite"
    d.mkdir()
    data = [
        {"company": "A", "bullets": ["foo bar"]},
        {"company": "B", "bullets": ["baz qux"]}
    ]
    file = d / "data.json"
    file.write_text(json.dumps(data))
    return str(file)

def test_indexing_creates_two_documents(sample_json):
    document_store = InMemoryDocumentStore()
    run_indexing(sample_json, document_store=document_store)
    assert document_store.count_documents() == 2
