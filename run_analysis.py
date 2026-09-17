"""
Standalone Customer Segmentation Analysis Runner
Trains and benchmarks multiple clustering algorithms on the Mall Customers dataset.
"""

import sys
import os
import pandas as pd
import numpy as np

from clustering_pipeline import (
    load_and_clean_data,
    get_feature_matrix,
    evaluate_clustering,
    train_kmeans,
    run_kmeans_elbow,
    train_hierarchical,
    compute_linkage_matrix,
    train_dbscan,
    compute_k_distances,
    train_gmm,
    run_gmm_evaluation,
    train_meanshift,
    train_spectral,
    get_cluster_profiles
)


def run_benchmark():
    print("=" * 80)
    print("       MALL CUSTOMER SEGMENTATION - MULTI-ALGORITHM BENCHMARK")
    print("=" * 80)
    
    # 1. Load Data
    data_path = "Mall_Customers.csv"
    if not os.path.exists(data_path):
        print(f"Error: Dataset file '{data_path}' not found.")
        return
        
    df = load_and_clean_data(data_path)
    print(f"\n[+] Loaded Dataset successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"    - Gender Distribution: {dict(df['gender'].value_counts())}")
    print(f"    - Age Range: {df['age'].min()} - {df['age'].max()} (Mean: {df['age'].mean():.1f})")
    print(f"    - Annual Income Range: ${df['annual_income'].min()}k - ${df['annual_income'].max()}k (Mean: ${df['annual_income'].mean():.1f}k)")
    print(f"    - Spending Score Range: {df['spending_score'].min()} - {df['spending_score'].max()} (Mean: {df['spending_score'].mean():.1f})")

    # 2. Benchmark Feature Set: Annual Income & Spending Score (Standard 2D Benchmark)
    feature_sets = {
        "2D (Income & Spending Score)": ["annual_income", "spending_score"],
        "3D (Age, Income & Spending Score)": ["age", "annual_income", "spending_score"],
        "4D (Gender, Age, Income & Spending Score)": ["gender_encoded", "age", "annual_income", "spending_score"]
    }

    all_benchmark_results = []
    
    for feat_name, features in feature_sets.items():
        print("\n" + "-" * 80)
        print(f" FEATURE SET: {feat_name}")
        print("-" * 80)
        
        X_scaled, scaler = get_feature_matrix(df, features, scaler_type="standard")
        
        # Algorithms to run
        algorithms = {}
        
        # K-Means (k=5)
        km_model, km_labels, km_metrics = train_kmeans(X_scaled, n_clusters=5)
        algorithms["K-Means (k=5)"] = (km_labels, km_metrics)
        
        # Hierarchical - Ward (k=5)
        agg_ward_model, agg_ward_labels, agg_ward_metrics = train_hierarchical(X_scaled, n_clusters=5, linkage_method="ward")
        algorithms["Hierarchical (Ward, k=5)"] = (agg_ward_labels, agg_ward_metrics)
        
        # Hierarchical - Complete (k=5)
        agg_comp_model, agg_comp_labels, agg_comp_metrics = train_hierarchical(X_scaled, n_clusters=5, linkage_method="complete")
        algorithms["Hierarchical (Complete, k=5)"] = (agg_comp_labels, agg_comp_metrics)
        
        # DBSCAN
        # Tune eps based on feature dimensionality
        eps_val = 0.45 if len(features) == 2 else 0.8
        db_model, db_labels, db_metrics = train_dbscan(X_scaled, eps=eps_val, min_samples=4)
        algorithms[f"DBSCAN (eps={eps_val})"] = (db_labels, db_metrics)
        
        # Gaussian Mixture Models (k=5)
        gmm_model, gmm_labels, gmm_metrics = train_gmm(X_scaled, n_components=5)
        algorithms["Gaussian Mixture (GMM, k=5)"] = (gmm_labels, gmm_metrics)
        
        # Mean Shift
        ms_model, ms_labels, ms_metrics = train_meanshift(X_scaled, quantile=0.25)
        algorithms["Mean Shift"] = (ms_labels, ms_metrics)
        
        # Spectral Clustering (k=5)
        try:
            sp_model, sp_labels, sp_metrics = train_spectral(X_scaled, n_clusters=5)
            algorithms["Spectral Clustering (k=5)"] = (sp_labels, sp_metrics)
        except Exception as e:
            algorithms["Spectral Clustering (k=5)"] = (np.zeros(len(df)), {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None, "n_clusters": 1, "n_noise": 0})
            
        print(f"{'Algorithm':<30} | {'Clusters':<8} | {'Noise':<6} | {'Silhouette':<12} | {'Davies-Bouldin':<15} | {'Calinski-Harabasz':<18}")
        print("-" * 105)
        for algo_name, (labels, metrics) in algorithms.items():
            sil_str = f"{metrics['silhouette']:.4f}" if metrics['silhouette'] is not None else "N/A"
            db_str = f"{metrics['davies_bouldin']:.4f}" if metrics['davies_bouldin'] is not None else "N/A"
            ch_str = f"{metrics['calinski_harabasz']:.2f}" if metrics['calinski_harabasz'] is not None else "N/A"
            print(f"{algo_name:<30} | {metrics['n_clusters']:<8} | {metrics['n_noise']:<6} | {sil_str:<12} | {db_str:<15} | {ch_str:<18}")
            
            all_benchmark_results.append({
                "Feature Set": feat_name,
                "Algorithm": algo_name,
                "Clusters": metrics["n_clusters"],
                "Noise Points": metrics["n_noise"],
                "Silhouette Score": metrics["silhouette"],
                "Davies-Bouldin Index": metrics["davies_bouldin"],
                "Calinski-Harabasz Score": metrics["calinski_harabasz"]
            })

    # 3. Detailed Cluster Profile for Primary 2D K-Means (Gold Standard)
    print("\n" + "=" * 80)
    print(" DETAILED BUSINESS SEGMENT PERSONAS (K-Means, Income & Spending)")
    print("=" * 80)
    
    X_std, _ = get_feature_matrix(df, ["annual_income", "spending_score"], scaler_type="standard")
    best_km, best_labels, _ = train_kmeans(X_std, n_clusters=5)
    profiles_df = get_cluster_profiles(df, best_labels)
    
    for _, row in profiles_df.iterrows():
        print(f"\n* [{row['Cluster ID']}] {row['Persona']} ({row['Share (%)']}% of customers, N={row['Count']})")
        print(f"  Traits        : {row['Key Traits']} | {row['Age Bracket']}")
        print(f"  Averages      : Age: {row['Avg Age']} yrs | Income: ${row['Avg Income ($k)']}k | Spending Score: {row['Avg Spending Score']}/100")
        print(f"  Demographics  : {row['Female (%)']}% Female, {row['Male (%)']}% Male")
        print(f"  Strategy      : {row['Marketing Strategy']}")
        print(f"  Best Channels : {row['Recommended Channel']}")
        
    # 4. Save Clustered Dataset and Benchmark Leaderboard
    df_clustered = df.copy()
    df_clustered["cluster_kmeans_2d"] = best_labels
    df_clustered["persona"] = df_clustered["cluster_kmeans_2d"].map(dict(zip(range(len(profiles_df)), profiles_df["Persona"])))
    
    df_clustered.to_csv("mall_customers_segmented.csv", index=False)
    benchmark_df = pd.DataFrame(all_benchmark_results)
    benchmark_df.to_csv("clustering_benchmark_results.csv", index=False)
    
    print("\n" + "=" * 80)
    print(f"[OK] Analysis complete! Saved segmented dataset to 'mall_customers_segmented.csv'")
    print(f"[OK] Saved benchmark metrics to 'clustering_benchmark_results.csv'")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
