"""
rag_pipeline.py
---------------
In-memory vector search for GridWatch policy documents.

Embeddings: sentence-transformers/all-MiniLM-L6-v2
Retrieval:  cosine similarity via numpy / scikit-learn
Index:      cached in data/vector_index.pkl; rebuilt if missing
"""

import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── paths ──────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent.parent
INDEX_PATH    = ROOT_DIR / "data" / "vector_index.pkl"
POLICY_FOLDER = ROOT_DIR / "data" / "policy"

# ── embedding model (singleton) ────────────────────────────────────────────
_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ── module-level index cache ───────────────────────────────────────────────
# Structure: {"embeddings": np.ndarray, "documents": list, "metadatas": list}
_INDEX = None


# ── helpers ────────────────────────────────────────────────────────────────

def _chunk_by_paragraph(text: str, min_len: int = 60) -> list[str]:
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if len(p.strip()) >= min_len]


def _embed(texts: list[str]) -> np.ndarray:
    return _MODEL.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def load_csv_data(csv_path: Path) -> tuple[list, list]:
    df = pd.read_csv(csv_path)

    def _clean(val) -> str:
        if isinstance(val, str):
            return val.replace("$", "").replace(",", "").strip()
        return str(val)

    documents, metadatas = [], []
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
        metadatas.append({"source": "liheap_california.csv", "year": year, "type": "liheap_csv"})

    return documents, metadatas


# ── index build ────────────────────────────────────────────────────────────

def build_index(policy_folder: Path = POLICY_FOLDER) -> dict:
    """
    Builds (or loads from pickle) the in-memory vector index.
    Returns a dict with keys: embeddings, documents, metadatas.
    """
    global _INDEX
    if _INDEX is not None:
        return _INDEX

    if INDEX_PATH.exists():
        with open(INDEX_PATH, "rb") as f:
            _INDEX = pickle.load(f)
        return _INDEX

    documents, metadatas = [], []

    for txt_file in sorted(Path(policy_folder).glob("*.txt")):
        text   = txt_file.read_text(encoding="utf-8").strip()
        chunks = _chunk_by_paragraph(text)
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                "source":   txt_file.name,
                "chunk_id": i,
                "type":     "policy_text",
            })

    csv_path = Path(policy_folder) / "liheap_california.csv"
    if csv_path.exists():
        csv_docs, csv_metas = load_csv_data(csv_path)
        documents.extend(csv_docs)
        metadatas.extend(csv_metas)

    embeddings = _embed(documents)

    _INDEX = {"embeddings": embeddings, "documents": documents, "metadatas": metadatas}

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(_INDEX, f)

    return _INDEX


# ── retrieval ──────────────────────────────────────────────────────────────

def retrieve_context(
    query: str,
    city_data: dict | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Semantic search over the in-memory vector index.

    Args:
        query:     The user's question or SHAP-derived query string.
        city_data: Optional dict with city profile keys
                   (energy_burden, household_income, avg_annual_energy_cost).
        top_k:     Number of chunks to return.

    Returns:
        List of dicts: {text, source, type, score}
        where score is cosine similarity (0–1, higher = more relevant).
    """
    index = build_index()
    if not index["documents"]:
        return []

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

    query_vec  = _embed([enriched])
    scores     = cosine_similarity(query_vec, index["embeddings"])[0]
    k          = min(top_k, len(scores))
    top_idx    = np.argsort(scores)[::-1][:k]

    return [
        {
            "text":   index["documents"][i],
            "source": index["metadatas"][i].get("source", "unknown"),
            "type":   index["metadatas"][i].get("type", "unknown"),
            "score":  round(float(scores[i]), 3),
        }
        for i in top_idx
    ]


# ── quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building / loading index...")
    idx = build_index()
    print(f"Index has {len(idx['documents'])} chunks.\n")

    results = retrieve_context(
        query="low income heating assistance eligibility",
        city_data={"household_income": 45000, "energy_burden": 4.2, "avg_annual_energy_cost": 1800},
        top_k=3,
    )
    for i, r in enumerate(results, 1):
        print(f"[{i}] source={r['source']}  score={r['score']}")
        print(f"     {r['text'][:200]}\n")
