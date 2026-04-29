import pandas as pd

from policy_recommendation_from_shap import (
    load_policy_docs,
    build_chunk_index,
    get_top_shap_features_for_city,
    shap_to_query,
    retrieve_top_chunks,
    generate_grounded_recommendation,
    extract_program_names
)


def main():
    shap_path = "../data/processed/shap_values_50.csv"
    output_path = "../data/processed/policy_recommendations_50.csv"

    # Load SHAP file
    shap_df = pd.read_csv(shap_path)
    cities = shap_df["city"].dropna().unique().tolist()

    # Load policy docs once
    docs = load_policy_docs("../data/policy")
    chunk_records = build_chunk_index(docs)

    results = []

    for city_name in cities:
        try:
            shap_feature_dict = get_top_shap_features_for_city(
                shap_csv_path=shap_path,
                city_name=city_name,
                top_n=3
            )

            query = shap_to_query(shap_feature_dict)
            retrieved_chunks = retrieve_top_chunks(query, chunk_records, top_k=2)
            recommendation = generate_grounded_recommendation(
                city=city_name,
                shap_feature_dict=shap_feature_dict,
                retrieved_chunks=retrieved_chunks
            )

            program_names = extract_program_names(retrieved_chunks)

            results.append({
                "city": city_name,
                "top_feature_1": list(shap_feature_dict.keys())[0],
                "top_feature_1_value": list(shap_feature_dict.values())[0],
                "top_feature_2": list(shap_feature_dict.keys())[1],
                "top_feature_2_value": list(shap_feature_dict.values())[1],
                "top_feature_3": list(shap_feature_dict.keys())[2],
                "top_feature_3_value": list(shap_feature_dict.values())[2],
                "query": query,
                "retrieved_programs": "; ".join(program_names),
                "recommendation": recommendation
            })

        except Exception as e:
            results.append({
                "city": city_name,
                "top_feature_1": None,
                "top_feature_1_value": None,
                "top_feature_2": None,
                "top_feature_2_value": None,
                "top_feature_3": None,
                "top_feature_3_value": None,
                "query": None,
                "retrieved_programs": None,
                "recommendation": f"ERROR: {str(e)}"
            })

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_path, index=False)

    print(f"Saved batch recommendations to: {output_path}")
    print(output_df.head(10))


if __name__ == "__main__":
    main()