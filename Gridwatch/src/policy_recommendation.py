from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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

    return chunk_records


def retrieve_top_chunks(query, chunk_records, top_k=2):
    texts = [record["text"] for record in chunk_records]

    vectorizer = TfidfVectorizer(stop_words="english")
    chunk_vectors = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(query_vector, chunk_vectors).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "score": float(scores[idx]),
            "filename": chunk_records[idx]["filename"],
            "text": chunk_records[idx]["text"]
        })

    return results


def shap_to_query(shap_feature_dict):
    feature_phrases = {
        "energy_burden": "high energy burden",
        "household_income": "low household income",
        "avg_annual_energy_cost": "high annual energy cost",
        "avg_temp_forecast": "high forecast temperature",
        "max_temp_forecast": "extreme heat",
        "min_temp_forecast": "unusual overnight temperatures"
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
        return "energy assistance for low-income households"

    return "Programs for " + ", ".join(phrases)


def generate_grounded_recommendation(city, risk_label, shap_feature_dict, retrieved_chunks):
    top_features = sorted(
        shap_feature_dict.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]

    feature_names = [feature for feature, _ in top_features]

    feature_text_map = {
        "energy_burden": "high energy burden",
        "household_income": "low household income",
        "avg_annual_energy_cost": "high annual energy cost",
        "avg_temp_forecast": "high forecast temperature",
        "max_temp_forecast": "extreme heat",
        "min_temp_forecast": "unusual overnight temperatures"
    }

    readable_features = [
        feature_text_map.get(feature, feature.replace("_", " "))
        for feature in feature_names
    ]

    context_summary = " ".join([chunk["text"] for chunk in retrieved_chunks])

    recommendation = (
        f"{city} is classified as {'high risk' if risk_label == 1 else 'lower risk'} "
        f"based on factors such as {', '.join(readable_features)}. "
        f"According to the retrieved policy context, assistance programs like LIHEAP may help "
        f"households manage heating and cooling costs, reduce immediate utility bill pressure, "
        f"and provide crisis support during extreme weather conditions. "
        f"Grounding context: {context_summary}"
    )

    return recommendation


def main():
    docs = load_policy_docs("../data/policy")
    chunk_records = build_chunk_index(docs)

    shap_example = {
        "energy_burden": 0.42,
        "max_temp_forecast": 0.31,
        "household_income": 0.28
    }

    city = "Adelanto"
    risk_label = 1

    query = shap_to_query(shap_example)
    retrieved_chunks = retrieve_top_chunks(query, chunk_records, top_k=2)

    recommendation = generate_grounded_recommendation(
        city=city,
        risk_label=risk_label,
        shap_feature_dict=shap_example,
        retrieved_chunks=retrieved_chunks
    )

    print("\nGenerated Query:\n")
    print(query)

    print("\nRetrieved Chunks:\n")
    for chunk in retrieved_chunks:
        print(f"File: {chunk['filename']}")
        print(f"Score: {chunk['score']:.4f}")
        print(f"Text: {chunk['text']}")
        print("-" * 80)

    print("\nGrounded Recommendation:\n")
    print(recommendation)


if __name__ == "__main__":
    main()