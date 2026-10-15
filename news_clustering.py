"""
Project 2: News Article Clustering System using TF-IDF
MCA201P Lab Sheet-07 | Python 3.11+
"""
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import cosine_distances

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "news_articles.csv"
OUTPUT_DIR = ROOT / "outputs"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    data = pd.read_csv(DATA_FILE)
    required = {"Article_ID", "Text"}
    if not required.issubset(data.columns):
        raise ValueError(f"Dataset must contain columns: {sorted(required)}")
    data["Text"] = data["Text"].fillna("").astype(str).str.strip()
    data = data.drop_duplicates(subset=["Text"])
    data = data[data["Text"].str.len() > 0].reset_index(drop=True)
    if len(data) < 3:
        raise ValueError("At least 3 non-empty articles are required.")

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.95, min_df=1,
                                 ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(data["Text"])
    if matrix.shape[1] < 2:
        raise ValueError("Not enough distinct text features to cluster the articles.")

    max_k = min(8, len(data) - 1)
    scores = {}
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(matrix)
        if len(set(labels)) > 1:
            scores[k] = silhouette_score(matrix, labels, metric="cosine")
    if not scores:
        raise RuntimeError("Could not calculate a silhouette score for candidate clusters.")
    best_k = max(scores, key=scores.get)

    km_labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(matrix)
    # Average-linkage hierarchical clustering uses cosine distance on TF-IDF vectors.
    distances = cosine_distances(matrix)
    ag_labels = AgglomerativeClustering(n_clusters=best_k, metric="precomputed",
                                        linkage="average").fit_predict(distances)

    result = data.copy()
    result["KMeans_Cluster"] = km_labels
    result["Agglomerative_Cluster"] = ag_labels
    result.to_csv(OUTPUT_DIR / "news_cluster_assignments.csv", index=False)

    metrics = []
    for name, labels in [("K-Means (TF-IDF)", km_labels),
                         ("Agglomerative (cosine)", ag_labels)]:
        silhouette = silhouette_score(matrix, labels, metric="cosine") if len(set(labels)) > 1 else float("nan")
        row = {"Method": name, "Clusters": len(set(labels)),
               "Silhouette_Score_Cosine": round(float(silhouette), 4)}
        if "Category" in data.columns and data["Category"].nunique() > 1:
            row["Adjusted_Rand_Index_vs_Category"] = round(float(adjusted_rand_score(data["Category"], labels)), 4)
        metrics.append(row)
    pd.DataFrame(metrics).to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    # Save the highest-weight terms per cluster for interpretation.
    terms = np.array(vectorizer.get_feature_names_out())
    feature_weights = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(matrix).cluster_centers_
    term_rows = []
    for cluster_id, weights in enumerate(feature_weights):
        top_indices = np.argsort(weights)[::-1][:10]
        term_rows.append({"Cluster": cluster_id, "Top_Terms": ", ".join(terms[top_indices])})
    pd.DataFrame(term_rows).to_csv(OUTPUT_DIR / "cluster_top_terms.csv", index=False)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    plt.plot(list(scores.keys()), list(scores.values()), marker="o")
    plt.title("News Clustering: Silhouette Score by K")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Cosine silhouette score (higher is better)")
    plt.xticks(list(scores.keys()))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "silhouette_scores.png", dpi=160)
    plt.close()

    if matrix.shape[1] >= 2:
        svd = TruncatedSVD(n_components=2, random_state=42)
        coords = svd.fit_transform(matrix)
        plot_data = pd.DataFrame({"Component 1": coords[:, 0], "Component 2": coords[:, 1],
                                  "Cluster": km_labels.astype(str)})
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=plot_data, x="Component 1", y="Component 2",
                        hue="Cluster", palette="tab10", s=65)
        plt.title(f"News Article Clusters (K-Means, K={best_k}; TF-IDF + SVD)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "news_clusters_2d.png", dpi=160)
        plt.close()

    # A dendrogram is useful for visual analysis; use a bounded sample for readability.
    sample_n = min(30, len(data))
    if sample_n >= 3:
        sample_matrix = matrix[:sample_n]
        condensed = cosine_distances(sample_matrix)
        np.fill_diagonal(condensed, 0)
        hierarchy = linkage(condensed[np.triu_indices(sample_n, k=1)], method="average")
        plt.figure(figsize=(12, 6))
        dendrogram(hierarchy, labels=data["Article_ID"].iloc[:sample_n].tolist(),
                   leaf_rotation=90, leaf_font_size=8)
        plt.title("Hierarchical Clustering Dendrogram (first 30 articles maximum)")
        plt.ylabel("Cosine distance")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "hierarchical_dendrogram.png", dpi=160)
        plt.close()

    print("News article clustering completed successfully.")
    print(f"Valid articles: {len(data)}")
    print(f"TF-IDF feature count: {matrix.shape[1]}")
    print(f"Selected K: {best_k} (best cosine silhouette score among tested K values)")
    print("\nModel comparison:")
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
