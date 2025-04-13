import cmd
from pathlib import Path

import click
from haystack import Pipeline
from haystack.components.converters import MarkdownToDocument, TextFileToDocument
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

config = {"results": 10}

document_store = InMemoryDocumentStore()


def run_indexing(document_store=document_store):

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("converter", TextFileToDocument())
    # indexing_pipeline.add_component("cleaner", DocumentCleaner())
    indexing_pipeline.add_component(
        "splitter", DocumentSplitter(split_by="line", split_length=1)
    )
    indexing_pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder())
    indexing_pipeline.add_component(
        "writer", DocumentWriter(document_store=document_store)
    )
    indexing_pipeline.connect("converter", "splitter")
    indexing_pipeline.connect("splitter", "embedder")
    indexing_pipeline.connect("embedder", "writer")

    path = "instance/corpus"
    files = list(Path(path).glob("*.md"))

    print("Indexing files.")
    indexing_pipeline.run({"converter": {"sources": files}})


def run_query(query, document_store=document_store):
    ## Querying Pipeline
    query_pipeline = Pipeline()
    query_pipeline.add_component("embedder", SentenceTransformersTextEmbedder())
    query_pipeline.add_component(
        "retriever",
        InMemoryEmbeddingRetriever(
            document_store=document_store, scale_score=True, top_k=config["results"]
        ),
    )
    query_pipeline.connect("embedder.embedding", "retriever.query_embedding")

    return query_pipeline.run({"embedder": {"text": query}})["retriever"]["documents"]


def display_docs(docs):
    count = 0
    count_of = len(docs)
    for document in docs:
        count += 1
        md_number = click.style(f"- {count}.", fg="green")
        md_content = click.style(f"{document.content.rstrip()}")
        md_source = click.style(
            f"**(see: {document.meta['file_path']})**", italic=True, fg="blue"
        )
        click.echo(" ".join([md_number, md_content, md_source]), color=True)


class KiteShell(cmd.Cmd):
    intro = "Welcome to KITE, the Keyword Insight and Term Extraction Project\n\
        Type help or ? to list commands.\n"
    prompt = "(kite): "

    def do_index(self, arg):
        "Run the indexing pipeline"
        run_indexing(document_store=document_store)

    def do_query(self, arg):
        query = click.prompt(click.style("\nQUERY", fg="bright_white", bg="magenta"))
        docs = run_query(query, document_store=document_store)
        display_docs(docs)

    def do_quit(self, arg):
        "Quite KITE"
        return True


if __name__ == "__main__":
    KiteShell().cmdloop()
