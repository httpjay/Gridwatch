import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Singleton so the model is only loaded once per process
_embedding_model = None

def _get_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def load_policy_docs(folder_path):
    docs = []
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Policy folder not found: {folder_path}")

    for file_path in folder.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8").strip()
        if text:
            docs.append({
                "filename": file_path.name,
                "text": text
            })

    if not docs:
        raise ValueError(f"No .txt files found in: {folder_path}")

    return docs


def chunk_text(text, chunk_size=120, overlap=30):
    words = text.split()
    chunks = []

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


def build_chunk_index(docs):
    chunk_records = []

    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            chunk_records.append({
                "filename": doc["filename"],
                "chunk_id": i,
                "text": chunk
            })

    if not chunk_records:
        raise ValueError("No chunks were created from the policy files.")

    return chunk_records


def build_embedding_index(chunk_records):
    """
    Encodes all chunk texts into dense semantic embeddings using
    sentence-transformers (all-MiniLM-L6-v2).
    Returns (chunk_records, embeddings_matrix).
    """
    model = _get_model()
    texts = [r["text"] for r in chunk_records]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings


def retrieve_top_chunks(query, chunk_records, embeddings, top_k=3):
    """
    Semantic retrieval: encodes the query and ranks chunks by cosine similarity
    against precomputed embeddings.
    """
    model = _get_model()
    query_embedding = model.encode([query], convert_to_numpy=True)
    scores = cosine_similarity(query_embedding, embeddings).flatten()
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


def get_top_shap_features_for_city(shap_csv_path, city_name, top_n=3):
    df = pd.read_csv(shap_csv_path)

    if "city" not in df.columns:
        raise ValueError("Expected a 'city' column in shap_values_50.csv")

    city_row = df[df["city"].str.lower() == city_name.lower()]

    if city_row.empty:
        available_cities = sorted(df["city"].dropna().unique().tolist())
        raise ValueError(
            f"City '{city_name}' not found in SHAP file.\n"
            f"Available cities include: {available_cities[:10]}"
        )

    row = city_row.iloc[0]
    feature_cols = [col for col in df.columns if col != "city"]
    shap_dict = {feature: float(row[feature]) for feature in feature_cols}

    sorted_features = sorted(
        shap_dict.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return dict(sorted_features[:top_n])


def shap_to_query(shap_feature_dict):
    feature_phrases = {
        "household_income": "low household income energy assistance",
        "avg_annual_energy_cost": "high annual energy cost bill relief",
        "energy_burden": "high energy burden affordability",
        "avg_temp_forecast": "high forecast temperature cooling assistance",
        "max_temp_forecast": "extreme heat weatherization cooling",
        "min_temp_forecast": "cold overnight temperatures heating assistance"
    }

    phrases = []
    for feature_name in shap_feature_dict.keys():
        if feature_name in feature_phrases:
            phrases.append(feature_phrases[feature_name])

    if not phrases:
        return "energy assistance programs for low-income households California"

    return " ".join(phrases)


def extract_program_names(retrieved_chunks):
    # Primary lookup: filename → short program name
    filename_to_program = {
        "care_fera.txt":          "CARE / FERA",
        "liheap_ladwp.txt":       "LIHEAP (LADWP)",
        "liheap_california.txt":  "LIHEAP",
        "esa_weatherization.txt": "ESA Program",
        "wap.txt":                "WAP",
    }

    program_names = []
    for chunk in retrieved_chunks:
        filename = chunk.get("filename", "")
        name = filename_to_program.get(filename)

        # Fallback: derive a short name from the filename itself
        if not name:
            stem = filename.replace(".txt", "").replace("_", " ").title()
            name = stem if stem else None

        if name and name not in program_names:
            program_names.append(name)

    return program_names


def generate_grounded_recommendation(city, shap_feature_dict, retrieved_chunks,
                                      llm_client=None, model_name=None):
    """
    Generates a policy recommendation for the given city.

    If llm_client is provided, passes the retrieved policy chunks to the LLM
    and asks it to write a specific, grounded recommendation (real RAG).
    Otherwise falls back to a template string.
    """
    feature_label_map = {
        "household_income": "low household income",
        "avg_annual_energy_cost": "high annual energy cost",
        "energy_burden": "high energy burden",
        "avg_temp_forecast": "high forecast temperature",
        "max_temp_forecast": "extreme heat / high maximum temperature",
        "min_temp_forecast": "unusual overnight temperatures"
    }

    readable_features = [
        feature_label_map.get(f, f.replace("_", " "))
        for f in shap_feature_dict.keys()
    ]

    if llm_client is None or model_name is None:
        # Fallback template (used by batch scripts without an LLM client)
        program_names = extract_program_names(retrieved_chunks)
        if not program_names:
            program_names = ["energy assistance programs"]
        program_text = (
            program_names[0] if len(program_names) == 1
            else " and ".join(program_names) if len(program_names) == 2
            else ", ".join(program_names[:-1]) + f", and {program_names[-1]}"
        )
        return (
            f"For {city}, the most influential SHAP factors include "
            f"{', '.join(readable_features)}. "
            f"Based on the retrieved policy context, programs such as {program_text} may help reduce "
            f"energy bill pressure, support income-qualified households, and improve resilience during "
            f"periods of temperature-related energy stress."
        )

    # --- Real RAG: LLM reads the actual retrieved policy text ---
    policy_context = "\n\n".join([
        f"[Source: {c['filename']}]\n{c['text']}"
        for c in retrieved_chunks
    ])

    system_prompt = (
        "You are an energy policy advisor specializing in California low-income assistance programs. "
        "Your recommendations must be grounded strictly in the policy excerpts provided. "
        "Do not invent program names, eligibility rules, or dollar amounts not mentioned in the excerpts. "
        "Be specific, practical, and concise."
    )

    user_prompt = f"""A machine learning model (XGBoost + SHAP) flagged {city} as high energy affordability risk.

Top risk drivers identified by the model:
{chr(10).join(f'- {f}' for f in readable_features)}

Relevant policy program excerpts retrieved from California assistance program documents:
{policy_context}

Write a 3–4 sentence recommendation for {city} that:
1. Names the specific programs from the excerpts that best match this city's risk factors
2. Explains why those programs address the specific drivers listed above
3. Mentions any eligibility criteria or next steps found in the excerpts
"""

    response = llm_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()


def main():
    if len(sys.argv) > 1:
        city_name = " ".join(sys.argv[1:])
    else:
        city_name = "Adelanto"

    docs = load_policy_docs("../data/policy")
    chunk_records = build_chunk_index(docs)
    embeddings = build_embedding_index(chunk_records)

    shap_feature_dict = get_top_shap_features_for_city(
        shap_csv_path="../data/processed/shap_values_50.csv",
        city_name=city_name,
        top_n=3
    )

    query = shap_to_query(shap_feature_dict)
    retrieved_chunks = retrieve_top_chunks(query, chunk_records, embeddings, top_k=3)

    recommendation = generate_grounded_recommendation(
        city=city_name,
        shap_feature_dict=shap_feature_dict,
        retrieved_chunks=retrieved_chunks
        # no llm_client → uses fallback template in standalone mode
    )

    print(f"\nTop SHAP Features for {city_name}:\n")
    for feature, value in shap_feature_dict.items():
        print(f"  {feature}: {value:.6f}")

    print("\nGenerated Semantic Query:\n")
    print(f"  {query}")

    print("\nRetrieved Chunks:\n")
    for chunk in retrieved_chunks:
        print(f"  File: {chunk['filename']}  |  Score: {chunk['score']:.4f}")
        print(f"  Text: {chunk['text'][:200]}...")
        print("-" * 80)

    print("\nGrounded Recommendation:\n")
    print(recommendation)


if __name__ == "__main__":
    main()
