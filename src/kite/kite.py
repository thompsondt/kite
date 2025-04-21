import cmd
import json

import click
from haystack import Document, Pipeline
from haystack.components.converters import MarkdownToDocument, TextFileToDocument
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

config = {"results": 10}

document_store = InMemoryDocumentStore()


def run_indexing(src, document_store=document_store):

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder())
    indexing_pipeline.add_component(
        "writer", DocumentWriter(document_store=document_store)
    )
    indexing_pipeline.connect("embedder", "writer")

    def get_experiences(experience={}):
        documents = []
        if experience.get("bullets") is None:
            return documents
        bullets = experience.pop("bullets")
        for bullet in bullets:
            documents.append(Document(content=bullet, meta=experience))
        return documents

    def get_documents():

        documents = []
        with open(src) as file:
            experience = json.load(file)
        for item in experience:
            documents = documents + get_experiences(item)
        return documents

    documents = get_documents()
    print(f"Indexing {len(documents)} documents.")
    indexing_pipeline.run({"embedder": {"documents": documents}})


def run_query(query, document_store=document_store):
    ## Querying Pipeline
    query_pipeline = Pipeline()
    query_pipeline.add_component("document_joiner", DocumentJoiner())
    query_pipeline.add_component("ranker", TransformersSimilarityRanker())
    query_pipeline.add_component(
        "bm25_retriever", InMemoryBM25Retriever(document_store=document_store)
    )
    query_pipeline.add_component("embedder", SentenceTransformersTextEmbedder())
    query_pipeline.add_component(
        "embedding_retriever",
        InMemoryEmbeddingRetriever(
            document_store=document_store, scale_score=True, top_k=config["results"]
        ),
    )
    query_pipeline.connect("embedder.embedding", "embedding_retriever.query_embedding")
    query_pipeline.connect("bm25_retriever", "document_joiner")
    query_pipeline.connect("embedding_retriever", "document_joiner")
    query_pipeline.connect("document_joiner", "ranker")

    return query_pipeline.run(
        {
            "embedder": {"text": query},
            "bm25_retriever": {"query": query},
            "ranker": {"query": query},
        }
    )["ranker"]["documents"]


def display_docs(docs):
    count = 0
    count_of = len(docs)
    for document in docs:
        count += 1
        md_number = click.style(f"- {count}.", fg="green")
        md_content = click.style(f"{document.content.rstrip()}")
        md_source = click.style(
            f"**(see: {document.meta.get("file_path",document.meta)})**",
            italic=True,
            fg="blue",
        )
        click.echo(" ".join([md_number, md_content, md_source]), color=True)


class KiteShell(cmd.Cmd):
    intro = "Welcome to KITE, the Keyword Insight and Term Extraction Project\n\
        Type help or ? to list commands.\n"
    prompt = "(kite): "

    def do_index(self, arg):
        "index data from a json data file\nUsage: index [SRC]"
        src = arg or click.prompt("JSON file path")
        try:
            run_indexing(src, document_store=document_store)
        except FileNotFoundError as e:
            print(e)

    def do_query(self, arg):
        query = click.prompt(click.style("\nQUERY", fg="bright_white", bg="magenta"))
        docs = run_query(query, document_store=document_store)
        display_docs(docs)

    def do_quit(self, arg):
        "Quite KITE"
        return True


if __name__ == "__main__":
    KiteShell().cmdloop()
