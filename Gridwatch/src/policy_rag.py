import os
import re
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_policy_docs(folder_path):
    docs = []
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Policy docs folder not found: {folder_path}")

    for file_path in folder.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8").strip()
        if text:
            docs.append({
                "filename": file_path.name,
                "text": text
            })

    if not docs:
        raise ValueError(f"No .txt policy documents found in: {folder_path}")

    return docs


def chunk_text(text, chunk_size=120, overlap=30):
    words = text.split()
    chunks = []

    if len(words) == 0:
        return chunks

    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break

        start += chunk_size - overlap

    return chunks


def build_chunk_index(docs, chunk_size=120, overlap=30):
    chunk_records = []

    for doc in docs:
        chunks = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, chunk in enumerate(chunks):
            chunk_records.append({
                "filename": doc["filename"],
                "chunk_id": i,
                "text": chunk
            })

    if not chunk_records:
        raise ValueError("No chunks were created from the policy documents.")

    return chunk_records


def retrieve_top_chunks(query, chunk_records, top_k=3):
    chunk_texts = [record["text"] for record in chunk_records]

    vectorizer = TfidfVectorizer(stop_words="english")
    chunk_vectors = vectorizer.fit_transform(chunk_texts)
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(query_vector, chunk_vectors).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "score": float(scores[idx]),
            "filename": chunk_records[idx]["filename"],
            "chunk_id": chunk_records[idx]["chunk_id"],
            "text": chunk_records[idx]["text"]
        })

    return results


def shap_to_query(shap_feature_dict):
    """
    Converts top SHAP features into a natural-language retrieval query.
    Example input:
    {
        "energy_burden": 0.42,
        "max_temp_forecast": 0.31,
        "household_income": 0.28
    }
    """
    feature_phrases = {
        "energy_burden": "high energy burden",
        "household_income": "low household income",
        "avg_annual_energy_cost": "high annual energy cost",
        "avg_temp_forecast": "high forecast temperature",
        "max_temp_forecast": "extreme heat or high maximum temperature",
        "min_temp_forecast": "unusual overnight temperature conditions"
    }

    sorted_features = sorted(
        shap_feature_dict.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    phrases = []
    for feature_name, _ in sorted_features[:3]:
        if feature_name in feature_phrases:
            phrases.append(feature_phrases[feature_name])

    if not phrases:
        return "energy assistance programs for low-income households"

    return "Programs related to " + ", ".join(phrases)


def main():
    # 1. Load policy docs
    docs = load_policy_docs("../data/policy")

    # 2. Chunk docs
    chunk_records = build_chunk_index(docs, chunk_size=120, overlap=30)

    print(f"Loaded {len(docs)} policy documents")
    print(f"Created {len(chunk_records)} chunks")

    # 3. Example SHAP-driven query
    shap_example = {
        "energy_burden": 0.42,
        "max_temp_forecast": 0.31,
        "household_income": 0.28
    }

    query = shap_to_query(shap_example)
    print("\nGenerated retrieval query:")
    print(query)

    # 4. Retrieve top chunks
    results = retrieve_top_chunks(query, chunk_records, top_k=3)

    print("\nTop retrieved policy chunks:\n")
    for i, result in enumerate(results, start=1):
        print(f"Result {i}")
        print(f"File: {result['filename']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['text']}")
        print("-" * 80)


if __name__ == "__main__":
    main()