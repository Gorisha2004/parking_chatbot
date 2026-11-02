"""Print the SQL tables and a Chroma chunk sample, to show the static/dynamic data split.

SQL holds dynamic data (working hours, prices, availability, reservations); Chroma holds
static data (the markdown docs, chunked). No LLM call: Chroma chunks are read directly
from the local collection, not found via a similarity search.
"""
from contextlib import closing

from parking_chatbot.db.database import get_connection
from parking_chatbot.rag.vector_store import get_vector_store

SAMPLE_CHUNKS = 3


def print_sql_tables() -> None:
    print("=== SQL (dynamic data): data/parking.db ===\n")
    with closing(get_connection()) as conn:
        tables = [
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        ]
        for table in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"{table} ({count} rows)")
            for row in conn.execute(f"SELECT * FROM {table} LIMIT 2"):
                print(f"  {row}")
            print()


def print_chroma_sample() -> None:
    print("=== Chroma (static data): chunked markdown docs ===\n")
    collection = get_vector_store()._collection
    print(f"Total chunks: {collection.count()}\n")
    sample = collection.get(limit=SAMPLE_CHUNKS, include=["documents", "metadatas"])
    for doc_id, text, metadata in zip(sample["ids"], sample["documents"], sample["metadatas"]):
        print(f"[{doc_id}] source={metadata.get('source')}")
        print(f"  {text[:150]}...")
        print()


if __name__ == "__main__":
    print_sql_tables()
    print_chroma_sample()
