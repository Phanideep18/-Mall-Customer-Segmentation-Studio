"""
Customer Segmentation ML Pipeline
Mall Customers Dataset - Comprehensive Clustering Suite
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, estimate_bandwidth, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram


def load_and_clean_data(file_path: str = "Mall_Customers.csv") -> pd.DataFrame:
    """
    Load Mall Customers dataset and standardize column names.
    """
    df = pd.read_csv(file_path)
    df.columns = [
        "customer_id",
        "gender",
        "age",
        "annual_income",
        "spending_score"
    ]
    # Add gender binary encoding for algorithms that require numeric features
    df["gender_encoded"] = df["gender"].map({"Male": 1, "Female": 0})
    return df


def get_feature_matrix(
    df: pd.DataFrame, 
    feature_cols: List[str], 
    scaler_type: str = "standard"
) -> Tuple[np.ndarray, Any]:
    """
    Extract features and scale them using specified scaler.
    """
    X_raw = df[feature_cols].values
    
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = None

    if scaler is not None:
        X_scaled = scaler.fit_transform(X_raw)
    else:
        X_scaled = X_raw.copy()
        
    return X_scaled, scaler


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> Dict[str, Optional[float]]:
    """
    Compute Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score.
    Handles noise points (-1) for density-based algorithms.
    """
    valid_mask = labels != -1
    unique_labels = set(labels[valid_mask])
    
    # Must have at least 2 distinct clusters and at least 2 points
    if len(unique_labels) < 2 or np.sum(valid_mask) < len(unique_labels) + 1:
        return {
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
            "n_clusters": len(unique_labels),
            "n_noise": int(np.sum(labels == -1))
        }
        
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]
    
    sil = float(silhouette_score(X_valid, labels_valid))
    db = float(davies_bouldin_score(X_valid, labels_valid))
    ch = float(calinski_harabasz_score(X_valid, labels_valid))
    
    return {
        "silhouette": round(sil, 4),
        "davies_bouldin": round(db, 4),
        "calinski_harabasz": round(ch, 2),
        "n_clusters": len(unique_labels),
        "n_noise": int(np.sum(labels == -1))
    }


# =====================================================================
# 1. K-MEANS CLUSTERING
# =====================================================================
def run_kmeans_elbow(X: np.ndarray, k_range: range = range(2, 11), random_state: int = 42) -> pd.DataFrame:
    """
    Computes WCSS (Inertia) and Silhouette score for a range of k values.
    """
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        metrics = evaluate_clustering(X, labels)
        results.append({
            "k": k,
            "wcss": km.inertia_,
            "silhouette": metrics["silhouette"],
            "davies_bouldin": metrics["davies_bouldin"],
            "calinski_harabasz": metrics["calinski_harabasz"]
        })
    return pd.DataFrame(results)


def train_kmeans(X: np.ndarray, n_clusters: int = 5, random_state: int = 42) -> Tuple[KMeans, np.ndarray, Dict[str, Any]]:
    """
    Fits K-Means model on dataset.
    """
    model = KMeans(n_clusters=n_clusters, init="k-means++", n_init=15, random_state=random_state)
    labels = model.fit_predict(X)
    metrics = evaluate_clustering(X, labels)
    return model, labels, metrics


# =====================================================================
# 2. HIERARCHICAL / AGGLOMERATIVE CLUSTERING
# =====================================================================
def compute_linkage_matrix(X: np.ndarray, method: str = "ward", metric: str = "euclidean") -> np.ndarray:
    """
    Computes hierarchical clustering linkage matrix for dendrogram visualization.
    """
    if method == "ward":
        metric = "euclidean"
    return linkage(X, method=method, metric=metric)


def train_hierarchical(
    X: np.ndarray, 
    n_clusters: int = 5, 
    linkage_method: str = "ward", 
    metric: str = "euclidean"
) -> Tuple[AgglomerativeClustering, np.ndarray, Dict[str, Any]]:
    """
    Fits Agglomerative Hierarchical Clustering model.
    """
    if linkage_method == "ward":
        metric = "euclidean"
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method, metric=metric)
    labels = model.fit_predict(X)
    metrics = evaluate_clustering(X, labels)
    return model, labels, metrics


# =====================================================================
# 3. DBSCAN (DENSITY-BASED SPATIAL CLUSTERING)
# =====================================================================
def compute_k_distances(X: np.ndarray, k: int = 4) -> np.ndarray:
    """
    Calculates k-nearest neighbor distances for finding optimal epsilon.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_dist = np.sort(distances[:, k - 1])
    return k_dist


def train_dbscan(
    X: np.ndarray, 
    eps: float = 0.5, 
    min_samples: int = 5, 
    metric: str = "euclidean"
) -> Tuple[DBSCAN, np.ndarray, Dict[str, Any]]:
    """
    Fits DBSCAN model.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = model.fit_predict(X)
    metrics = evaluate_clustering(X, labels)
    return model, labels, metrics


# =====================================================================
# 4. GAUSSIAN MIXTURE MODELS (GMM)
# =====================================================================
def run_gmm_evaluation(
    X: np.ndarray, 
    k_range: range = range(2, 11), 
    covariance_type: str = "full", 
    random_state: int = 42
) -> pd.DataFrame:
    """
    Evaluates GMM with varying components using AIC and BIC.
    """
    results = []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=random_state)
        labels = gmm.fit_predict(X)
        metrics = evaluate_clustering(X, labels)
        results.append({
            "k": k,
            "aic": gmm.aic(X),
            "bic": gmm.bic(X),
            "silhouette": metrics["silhouette"],
            "davies_bouldin": metrics["davies_bouldin"]
        })
    return pd.DataFrame(results)


def train_gmm(
    X: np.ndarray, 
    n_components: int = 5, 
    covariance_type: str = "full", 
    random_state: int = 42
) -> Tuple[GaussianMixture, np.ndarray, Dict[str, Any]]:
    """
    Fits Gaussian Mixture Model.
    """
    model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state)
    labels = model.fit_predict(X)
    metrics = evaluate_clustering(X, labels)
    return model, labels, metrics


# =====================================================================
# 5. MEAN SHIFT CLUSTERING
# =====================================================================
def train_meanshift(
    X: np.ndarray, 
    quantile: float = 0.25, 
    n_samples: Optional[int] = None
) -> Tuple[MeanShift, np.ndarray, Dict[str, Any]]:
    """
    Fits Mean Shift Clustering with estimated bandwidth.
    """
    bw = estimate_bandwidth(X, quantile=quantile, n_samples=n_samples)
    if bw is None or bw <= 0:
        bw = None
    model = MeanShift(bandwidth=bw, bin_seeding=True)
    labels = model.fit_predict(X)
    metrics = evaluate_clustering(X, labels)
    return model, labels, metrics


# =====================================================================
# 6. SPECTRAL CLUSTERING
# =====================================================================
def train_spectral(
    X: np.ndarray, 
    n_clusters: int = 5, 
    affinity: str = "rbf", 
    random_state: int = 42
) -> Tuple[SpectralClustering, np.ndarray, Dict[str, Any]]:
    """
    Fits Spectral Clustering.
    """
    model = SpectralClustering(
        n_clusters=n_clusters, 
        affinity=affinity, 
        assign_labels="kmeans", 
        random_state=random_state
    )
    labels = model.fit_predict(X)
    metrics = evaluate_clustering(X, labels)
    return model, labels, metrics


# =====================================================================
# PERSONA PROFILING & MARKETING STRATEGIES
# =====================================================================
def assign_business_persona(
    avg_income: float, 
    avg_spending: float, 
    avg_age: float,
    overall_income_median: float = 61.5,
    overall_spending_median: float = 50.0
) -> Dict[str, str]:
    """
    Assign marketing persona archetype, color tag, and actionable strategies based on cluster attributes.
    """
    # Check for Balanced/Middle cluster first (Income around 40-75, Spending around 35-65)
    if 40 <= avg_income <= 75 and 35 <= avg_spending <= 65:
        persona = "Balanced Mainstream (Standard)"
        badge_color = "#3B82F6"  # Blue
        traits = "Moderate Income & Moderate Spending"
        description = "Steady, reliable middle-market shoppers with consistent repeat visits and balanced lifestyle spending."
        marketing_action = "Cross-category loyalty rewards, seasonal holiday bundles, subscription membership trials, cashback incentives."
        channel = "Omnichannel emails, mobile app push notifications, weekly promotional circulars."
    elif avg_income >= overall_income_median and avg_spending >= overall_spending_median:
        persona = "VIP Champions (High-Value Target)"
        badge_color = "#10B981"  # Emerald
        traits = "High Income & High Spending"
        description = "Affluent and highly engaged customers. Premium luxury drivers with highest customer lifetime value (CLV)."
        marketing_action = "Exclusive VIP private previews, luxury concierge services, early access to new collections, loyalty tier privileges."
        channel = "Dedicated relationship manager, private invitations, exclusive concierge app."
    elif avg_income >= overall_income_median and avg_spending < overall_spending_median:
        persona = "Careful Wealthy (Discerning Spenders)"
        badge_color = "#6366F1"  # Indigo
        traits = "High Income & Low/Moderate Spending"
        description = "High purchasing power with frugal, selective, or value-driven spending behavior."
        marketing_action = "Value-proposition highlights, premium quality and durability warranties, financial wellness perks, bundled utility packs."
        channel = "Curated email digests, premium webinars, targeted LinkedIn / professional network ads."
    elif avg_income < overall_income_median and avg_spending >= overall_spending_median:
        persona = "Trendsetting Spenders (High Engagement)"
        badge_color = "#F59E0B"  # Amber
        traits = "Modest Income & High Spending"
        description = "Impulsive, style-conscious, trend-sensitive buyers often younger in age and active on social media."
        marketing_action = "Flash sales, buy-now-pay-later (BNPL) financing, limited edition streetwear drops, influencer collaborations."
        channel = "Instagram, TikTok, real-time push alerts, influencer live streams."
    else:
        persona = "Budget Pragmatists (Sensible Essentials)"
        badge_color = "#64748B"  # Slate
        traits = "Low Income & Low Spending"
        description = "Cost-conscious shoppers focusing strictly on daily essentials and steep discount promotions."
        marketing_action = "Clearance alerts, essential value packs, price-match guarantees, bulk economy discounts."
        channel = "SMS alerts, discount circulars, localized coupon apps."

    if avg_age <= 30:
        age_tag = "Youth / Gen-Z & Millennial Focus"
    elif avg_age >= 50:
        age_tag = "Mature / Senior Demographic"
    else:
        age_tag = "Mid-Career Demographic"

    return {
        "persona_name": persona,
        "badge_color": badge_color,
        "traits": traits,
        "age_tag": age_tag,
        "description": description,
        "marketing_action": marketing_action,
        "recommended_channel": channel
    }


def get_cluster_profiles(
    df: pd.DataFrame, 
    labels: np.ndarray
) -> pd.DataFrame:
    """
    Generate comprehensive cluster summary statistics and business personas.
    """
    df_eval = df.copy()
    df_eval["cluster"] = labels
    
    # Filter out noise if any (-1)
    df_valid = df_eval[df_eval["cluster"] != -1]
    
    profiles = []
    overall_inc_med = df["annual_income"].median()
    overall_spn_med = df["spending_score"].median()
    
    for c in sorted(df_valid["cluster"].unique()):
        sub = df_valid[df_valid["cluster"] == c]
        count = len(sub)
        pct = (count / len(df)) * 100
        
        avg_age = sub["age"].mean()
        avg_income = sub["annual_income"].mean()
        avg_spending = sub["spending_score"].mean()
        male_pct = (sub["gender"] == "Male").mean() * 100
        female_pct = (sub["gender"] == "Female").mean() * 100
        
        persona_info = assign_business_persona(
            avg_income=avg_income,
            avg_spending=avg_spending,
            avg_age=avg_age,
            overall_income_median=overall_inc_med,
            overall_spending_median=overall_spn_med
        )
        
        profiles.append({
            "Cluster ID": f"Cluster {c}",
            "Persona": persona_info["persona_name"],
            "Count": count,
            "Share (%)": round(pct, 1),
            "Avg Age": round(avg_age, 1),
            "Avg Income ($k)": round(avg_income, 1),
            "Avg Spending Score": round(avg_spending, 1),
            "Female (%)": round(female_pct, 1),
            "Male (%)": round(male_pct, 1),
            "Key Traits": persona_info["traits"],
            "Age Bracket": persona_info["age_tag"],
            "Marketing Strategy": persona_info["marketing_action"],
            "Recommended Channel": persona_info["recommended_channel"],
            "Description": persona_info["description"],
            "Badge Color": persona_info["badge_color"]
        })
        
    return pd.DataFrame(profiles)


def predict_new_customer(
    customer_data: Dict[str, Any],
    model: Any,
    scaler: Any,
    feature_cols: List[str]
) -> Dict[str, Any]:
    """
    Assign a new customer input to a cluster and retrieve persona recommendations.
    """
    row_values = []
    for col in feature_cols:
        if col == "gender_encoded":
            val = 1 if customer_data.get("gender") == "Male" else 0
        else:
            val = customer_data.get(col, 0)
        row_values.append(val)
        
    X_input = np.array([row_values])
    if scaler is not None:
        X_scaled = scaler.transform(X_input)
    else:
        X_scaled = X_input

    # Model prediction
    if hasattr(model, "predict"):
        assigned_cluster = int(model.predict(X_scaled)[0])
    elif hasattr(model, "cluster_centers_"):
        # Calculate nearest centroid distance
        dists = np.linalg.norm(model.cluster_centers_ - X_scaled, axis=1)
        assigned_cluster = int(np.argmin(dists))
    else:
        assigned_cluster = 0
        
    persona_info = assign_business_persona(
        avg_income=customer_data.get("annual_income", 50),
        avg_spending=customer_data.get("spending_score", 50),
        avg_age=customer_data.get("age", 35)
    )
    
    return {
        "assigned_cluster": assigned_cluster,
        "persona_info": persona_info
    }
