"""
Project 1: Customer Segmentation using K-Means Clustering
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
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "customer_data.csv"
OUTPUT_DIR = ROOT / "outputs"
FEATURES = ["Age", "Annual_Income", "Spending_Score", "Purchase_Frequency_Per_Month"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    data = pd.read_csv(DATA_FILE)
    required = ["Customer_ID"] + FEATURES
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    data = data.drop_duplicates(subset=["Customer_ID"]).copy()
    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=FEATURES).reset_index(drop=True)
    if len(data) < 3:
        raise ValueError("At least 3 valid customer rows are required.")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[FEATURES])

    # Select k by the best silhouette score over a safe candidate range.
    max_k = min(8, len(data) - 1)
    candidates = range(2, max_k + 1)
    scores = {}
    for k in candidates:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
        if len(set(labels)) > 1:
            scores[k] = silhouette_score(scaled, labels)
    if not scores:
        raise RuntimeError("Could not calculate a silhouette score for candidate clusters.")
    best_k = max(scores, key=scores.get)

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    data["KMeans_Cluster"] = kmeans.fit_predict(scaled)
    agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    data["Agglomerative_Cluster"] = agg.fit_predict(scaled)
    data.to_csv(OUTPUT_DIR / "customer_cluster_assignments.csv", index=False)

    profile = data.groupby("KMeans_Cluster")[FEATURES].mean().round(2)
    profile["Customer_Count"] = data.groupby("KMeans_Cluster")["Customer_ID"].count()
    profile.to_csv(OUTPUT_DIR / "customer_segment_profiles.csv")

    comparison = []
    for name, labels in [
        ("K-Means", data["KMeans_Cluster"].to_numpy()),
        ("Agglomerative (Ward)", data["Agglomerative_Cluster"].to_numpy()),
    ]:
        score = silhouette_score(scaled, labels) if len(set(labels)) > 1 else float("nan")
        comparison.append({"Method": name, "Clusters": len(set(labels)), "Silhouette_Score": round(float(score), 4)})
    pd.DataFrame(comparison).to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    plt.plot(list(scores.keys()), list(scores.values()), marker="o")
    plt.title("Choosing K with Silhouette Score")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Silhouette score (higher is better)")
    plt.xticks(list(scores.keys()))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "silhouette_scores.png", dpi=160)
    plt.close()

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)
    plot_data = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1],
                              "Cluster": data["KMeans_Cluster"].astype(str)})
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=plot_data, x="PC1", y="PC2", hue="Cluster", palette="tab10", s=65)
    plt.title(f"Customer Segments (K-Means, K={best_k})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "customer_clusters_pca.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.heatmap(profile[FEATURES], annot=True, fmt=".1f", cmap="YlGnBu")
    plt.title("Average Customer Features by K-Means Segment")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "segment_profile_heatmap.png", dpi=160)
    plt.close()

    print("Customer segmentation completed successfully.")
    print(f"Valid customers: {len(data)}")
    print(f"Selected K: {best_k} (best silhouette score among tested K values)")
    print("\nModel comparison:")
    print(pd.DataFrame(comparison).to_string(index=False))
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
