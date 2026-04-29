"""
rag_pipeline.py
---------------
ChromaDB-backed RAG pipeline for GridWatch policy documents.

Embedding model: sentence-transformers/all-MiniLM-L6-v2
  (same model already used in policy_recommendation_from_shap.py,
  no Anthropic API key required)

ChromaDB is persisted locally at data/chroma_db/ so embeddings are
computed only once; subsequent runs load from disk.
"""

import re
import pandas as pd
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ── paths ──────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parent.parent
CHROMA_PATH    = ROOT_DIR / "data" / "chroma_db"
POLICY_FOLDER  = ROOT_DIR / "data" / "policy"
COLLECTION_NAME = "gridwatch_policy"

# ── embedding function (singleton) ────────────────────────────────────────
_EF = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def _get_collection() -> chromadb.Collection:
    """Returns the persistent ChromaDB collection (creates it if missing)."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_EF,
        metadata={"hnsw:space": "cosine"},
    )


# ── document ingestion ─────────────────────────────────────────────────────

def _chunk_by_paragraph(text: str, min_len: int = 60) -> list[str]:
    """Splits a document into non-empty paragraph-sized chunks."""
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if len(p.strip()) >= min_len]


def load_csv_data(csv_path: Path) -> tuple[list, list, list]:
    """
    Converts each row of liheap_california.csv into a readable text chunk:
      "In [YEAR], California LIHEAP average heating benefit was $[X].
       Income limit for a 4-person household was $[Y].
       Total eligible households: [Z]."

    Returns (documents, ids, metadatas).
    """
    df = pd.read_csv(csv_path)

    def _clean(val) -> str:
        if isinstance(val, str):
            return val.replace("$", "").replace(",", "").strip()
        return str(val)

    documents, ids, metadatas = [], [], []

    for _, row in df.iterrows():
        year    = str(row.get("Fiscal Year", "Unknown")).strip('"').strip("'")
        heating = _clean(row.get("Average Benefits per Household - Heating", "N/A"))
        cooling = _clean(row.get("Average Benefits per Household - Cooling", "N/A"))
        crisis  = _clean(row.get("Average Benefits per Household - Year Round Crisis", "N/A"))
        total   = _clean(row.get("State Income-Eligible Households - Total", "N/A"))
        income  = _clean(row.get("State Maximum Income for a 4-Person Household - Heating", "N/A"))

        text = (
            f"In {year}, California LIHEAP average heating benefit was ${heating}. "
            f"Average cooling benefit was ${cooling}. "
            f"Average year-round crisis benefit was ${crisis}. "
            f"Income limit for a 4-person household was ${income}. "
            f"Total income-eligible households: {total}."
        )

        documents.append(text)
        ids.append(f"liheap_csv_{year}")
        metadatas.append({"source": "liheap_california.csv", "year": year, "type": "liheap_csv"})

    return documents, ids, metadatas


def build_index(policy_folder: Path = POLICY_FOLDER) -> chromadb.Collection:
    """
    Indexes all .txt policy files + liheap_california.csv into ChromaDB.
    Idempotent: if the collection already has documents, returns immediately.
    """
    collection = _get_collection()

    if collection.count() > 0:
        return collection   # already indexed — load from disk

    documents, ids, metadatas = [], [], []

    # ── .txt policy files ──
    for txt_file in sorted(Path(policy_folder).glob("*.txt")):
        text   = txt_file.read_text(encoding="utf-8").strip()
        chunks = _chunk_by_paragraph(text)
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            ids.append(f"{txt_file.stem}_{i}")
            metadatas.append({
                "source":   txt_file.name,
                "chunk_id": i,
                "type":     "policy_text",
            })

    # ── liheap CSV ──
    csv_path = Path(policy_folder) / "liheap_california.csv"
    if csv_path.exists():
        csv_docs, csv_ids, csv_metas = load_csv_data(csv_path)
        documents.extend(csv_docs)
        ids.extend(csv_ids)
        metadatas.extend(csv_metas)

    # Add in batches (ChromaDB recommends ≤ 5000 per call)
    batch = 100
    for start in range(0, len(documents), batch):
        collection.add(
            documents=documents[start : start + batch],
            ids=ids[start : start + batch],
            metadatas=metadatas[start : start + batch],
        )

    return collection


# ── retrieval ──────────────────────────────────────────────────────────────

def retrieve_context(
    query: str,
    city_data: dict | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Semantic search over the ChromaDB collection.

    Args:
        query:     The user's question or SHAP-derived query string.
        city_data: Optional dict with city profile keys
                   (energy_burden, household_income, avg_annual_energy_cost).
                   Used to enrich the query embedding.
        top_k:     Number of chunks to return.

    Returns:
        List of dicts: {text, source, type, score}
        where score is cosine similarity (0–1, higher = more relevant).
    """
    collection = _get_collection()

    # Enrich query with city risk signals using natural language so the
    # embedding stays in policy-document space (raw numbers dilute the match).
    if city_data:
        burden = city_data.get("energy_burden", 0)
        income = city_data.get("household_income", 0)
        risk_phrases = []
        if burden >= 3:
            risk_phrases.append("high energy burden low income household assistance")
        if income and income < 60000:
            risk_phrases.append("low income energy program eligibility")
        enriched = query + (" " + " ".join(risk_phrases) if risk_phrases else "")
    else:
        enriched = query

    results = collection.query(
        query_texts=[enriched],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":   doc,
            "source": meta.get("source", "unknown"),
            "type":   meta.get("type", "unknown"),
            "score":  round(1.0 - float(dist), 3),
        })

    return chunks


# ── quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building / loading ChromaDB index...")
    col = build_index()
    print(f"Collection has {col.count()} chunks.\n")

    results = retrieve_context(
        query="low income heating assistance eligibility",
        city_data={"household_income": 45000, "energy_burden": 4.2, "avg_annual_energy_cost": 1800},
        top_k=3,
    )

    for i, r in enumerate(results, 1):
        print(f"[{i}] source={r['source']}  score={r['score']}")
        print(f"     {r['text'][:200]}\n")
