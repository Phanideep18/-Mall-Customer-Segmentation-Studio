"""
Customer Segmentation Intelligence Dashboard
Multi-Algorithm Clustering on Mall Customers Dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

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
    get_cluster_profiles,
    assign_business_persona,
    predict_new_customer
)

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Studio | AI Clustering",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 24px 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-card {
        background: #ffffff;
        padding: 18px 20px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #10b981;
        font-weight: 500;
        margin-top: 2px;
    }
    
    .persona-card {
        background: #ffffff;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        padding: 22px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_dataset():
    return load_and_clean_data("Mall_Customers.csv")

df = get_dataset()

# Top Header
st.markdown("""
<div class="main-header">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 15px;">
        <div>
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em;">
                🛍️ Mall Customer Segmentation Studio
            </h1>
            <p style="margin: 6px 0 0 0; opacity: 0.85; font-size: 1.05rem;">
                Advanced Multi-Algorithm Unsupervised ML Suite • Persona Profiling • Targeted Marketing Playbooks
            </p>
        </div>
        <div>
            <span style="background: rgba(255,255,255,0.15); padding: 8px 16px; border-radius: 30px; font-weight: 600; font-size: 0.9rem; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.2);">
                ✨ 6 Clustering Algorithms Supported
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Top KPI Summary Cards
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Total Customers</div>
        <div class="metric-value">{:,}</div>
        <div class="metric-sub">✓ 100% Complete Records</div>
    </div>
    """.format(len(df)), unsafe_allow_html=True)

with kpi2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Gender Ratio</div>
        <div class="metric-value">{:.0f}% <span style="font-size: 1rem; color: #ec4899;">♀</span> / {:.0f}% <span style="font-size: 1rem; color: #3b82f6;">♂</span></div>
        <div class="metric-sub">Female Predominant</div>
    </div>
    """.format((df['gender']=='Female').mean()*100, (df['gender']=='Male').mean()*100), unsafe_allow_html=True)

with kpi3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Avg Annual Income</div>
        <div class="metric-value">${:.1f}k</div>
        <div class="metric-sub">Range: ${}k - ${}k</div>
    </div>
    """.format(df['annual_income'].mean(), df['annual_income'].min(), df['annual_income'].max()), unsafe_allow_html=True)

with kpi4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Avg Spending Score</div>
        <div class="metric-value">{:.1f}<span style="font-size: 1rem; color: #64748b;">/100</span></div>
        <div class="metric-sub">Median: {:.0f} pts</div>
    </div>
    """.format(df['spending_score'].mean(), df['spending_score'].median()), unsafe_allow_html=True)

with kpi5:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Avg Customer Age</div>
        <div class="metric-value">{:.1f} <span style="font-size: 1rem; color: #64748b;">yrs</span></div>
        <div class="metric-sub">Range: {} - {} yrs</div>
    </div>
    """.format(df['age'].mean(), df['age'].min(), df['age'].max()), unsafe_allow_html=True)

st.write("")

# Navigation Tabs
tabs = st.tabs([
    "📊 Exploratory Data Analysis",
    "🔬 Clustering Studio & Model Lab",
    "🏆 Multi-Model Benchmark Leaderboard",
    "👥 Business Personas & Strategies",
    "🎯 Live Customer Segment Predictor",
    "📥 Data Explorer & Export"
])

# ==============================================================================
# TAB 1: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
with tabs[0]:
    st.subheader("Exploratory Data Analysis & Feature Distributions")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        eda_feature = st.selectbox(
            "Select Feature Distribution to Inspect",
            ["annual_income", "spending_score", "age"],
            format_func=lambda x: {
                "annual_income": "Annual Income ($k)",
                "spending_score": "Spending Score (1-100)",
                "age": "Age (Years)"
            }[x]
        )
        
        fig_dist = px.histogram(
            df, 
            x=eda_feature, 
            color="gender", 
            marginal="box",
            nbins=25,
            barmode="overlay",
            opacity=0.7,
            color_discrete_map={"Female": "#ec4899", "Male": "#3b82f6"},
            title=f"Distribution of {eda_feature.replace('_', ' ').title()} by Gender"
        )
        fig_dist.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", y=1.1, x=0.8))
        st.plotly_chart(fig_dist, use_container_width=True)
        
    with col2:
        fig_scatter_raw = px.scatter(
            df,
            x="annual_income",
            y="spending_score",
            color="gender",
            size="age",
            hover_data=["customer_id", "age", "gender"],
            color_discrete_map={"Female": "#ec4899", "Male": "#3b82f6"},
            title="Annual Income ($k) vs Spending Score (1-100) — Sized by Age",
            labels={"annual_income": "Annual Income ($k)", "spending_score": "Spending Score (1-100)"}
        )
        fig_scatter_raw.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", y=1.1, x=0.8))
        st.plotly_chart(fig_scatter_raw, use_container_width=True)
        
    col3, col4 = st.columns([1, 1])
    
    with col3:
        corr = df[["age", "annual_income", "spending_score", "gender_encoded"]].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="Viridis",
            labels=dict(color="Pearson Corr"),
            x=["Age", "Annual Income", "Spending Score", "Gender (Male=1)"],
            y=["Age", "Annual Income", "Spending Score", "Gender (Male=1)"],
            title="Feature Correlation Matrix"
        )
        fig_corr.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with col4:
        fig_3d_raw = px.scatter_3d(
            df,
            x="age",
            y="annual_income",
            z="spending_score",
            color="gender",
            color_discrete_map={"Female": "#ec4899", "Male": "#3b82f6"},
            title="3D Feature Space: Age × Income × Spending Score",
            labels={"age": "Age", "annual_income": "Income ($k)", "spending_score": "Spending Score"}
        )
        fig_3d_raw.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_3d_raw, use_container_width=True)


# ==============================================================================
# TAB 2: CLUSTERING STUDIO & MODEL LAB
# ==============================================================================
with tabs[1]:
    st.sidebar.markdown("### ⚙️ Clustering Configuration")
    
    feature_option = st.sidebar.selectbox(
        "Feature Set Dimension",
        [
            "2D: Annual Income & Spending Score",
            "3D: Age, Annual Income & Spending Score",
            "4D: Gender, Age, Annual Income & Spending Score"
        ]
    )
    
    if feature_option.startswith("2D"):
        selected_features = ["annual_income", "spending_score"]
    elif feature_option.startswith("3D"):
        selected_features = ["age", "annual_income", "spending_score"]
    else:
        selected_features = ["gender_encoded", "age", "annual_income", "spending_score"]
        
    scaler_choice = st.sidebar.selectbox(
        "Feature Scaling",
        ["standard", "robust", "minmax", "none"],
        format_func=lambda x: {
            "standard": "StandardScaler (Zero Mean, Unit Var)",
            "robust": "RobustScaler (Median / IQR)",
            "minmax": "MinMaxScaler (0 to 1)",
            "none": "No Scaling (Raw Values)"
        }[x]
    )
    
    algorithm_choice = st.sidebar.selectbox(
        "Clustering Algorithm",
        [
            "K-Means Clustering",
            "Hierarchical / Agglomerative",
            "DBSCAN (Density-Based)",
            "Gaussian Mixture Model (GMM)",
            "Mean Shift",
            "Spectral Clustering"
        ]
    )
    
    # Preprocess
    X_scaled, current_scaler = get_feature_matrix(
        df, 
        selected_features, 
        scaler_type=None if scaler_choice == "none" else scaler_choice
    )
    
    # Model specific options
    model_obj = None
    labels = None
    diag_fig = None
    
    if algorithm_choice == "K-Means Clustering":
        n_clusters = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=10, value=5)
        model_obj, labels, metrics = train_kmeans(X_scaled, n_clusters=n_clusters)
        
        # Diagnostic: Elbow Curve
        elbow_df = run_kmeans_elbow(X_scaled, k_range=range(2, 11))
        fig_elbow = make_subplots(specs=[[{"secondary_y": True}]])
        fig_elbow.add_trace(
            go.Scatter(x=elbow_df["k"], y=elbow_df["wcss"], name="WCSS (Inertia)", mode="lines+markers", line=dict(color="#6366f1", width=3)),
            secondary_y=False
        )
        fig_elbow.add_trace(
            go.Scatter(x=elbow_df["k"], y=elbow_df["silhouette"], name="Silhouette Score", mode="lines+markers", line=dict(color="#10b981", width=3, dash="dot")),
            secondary_y=True
        )
        fig_elbow.add_vline(x=n_clusters, line_dash="dash", line_color="#ef4444", annotation_text=f"Selected k={n_clusters}")
        fig_elbow.update_layout(title="K-Means Diagnostic: Elbow Method & Silhouette Score vs k", height=320, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", y=1.1, x=0.5))
        fig_elbow.update_xaxes(title_text="Number of Clusters (k)")
        fig_elbow.update_yaxes(title_text="WCSS (Inertia)", secondary_y=False)
        fig_elbow.update_yaxes(title_text="Silhouette Score", secondary_y=True)
        diag_fig = fig_elbow
        
    elif algorithm_choice == "Hierarchical / Agglomerative":
        n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=5)
        linkage_type = st.sidebar.selectbox("Linkage Criterion", ["ward", "complete", "average", "single"])
        metric_choice = "euclidean" if linkage_type == "ward" else st.sidebar.selectbox("Distance Metric", ["euclidean", "cosine", "cityblock"])
        
        model_obj, labels, metrics = train_hierarchical(X_scaled, n_clusters=n_clusters, linkage_method=linkage_type, metric=metric_choice)
        
    elif algorithm_choice == "DBSCAN (Density-Based)":
        eps_default = 0.45 if len(selected_features) == 2 else 0.8
        eps_val = st.sidebar.slider("Epsilon (Neighborhood Radius - ε)", min_value=0.1, max_value=2.0, value=eps_default, step=0.05)
        min_samples_val = st.sidebar.slider("Min Samples per Core Point", min_value=2, max_value=15, value=4)
        
        model_obj, labels, metrics = train_dbscan(X_scaled, eps=eps_val, min_samples=min_samples_val)
        
        # Diagnostic: k-distance graph
        k_dists = compute_k_distances(X_scaled, k=min_samples_val)
        fig_kdist = px.line(
            x=range(len(k_dists)), 
            y=k_dists,
            labels={"x": "Data Points (Sorted by Distance)", "y": f"{min_samples_val}-NN Distance"},
            title=f"DBSCAN Diagnostic: {min_samples_val}-NN Distance Graph for Epsilon (ε) Selection"
        )
        fig_kdist.add_hline(y=eps_val, line_dash="dash", line_color="#ef4444", annotation_text=f"Selected ε = {eps_val}")
        fig_kdist.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
        diag_fig = fig_kdist

    elif algorithm_choice == "Gaussian Mixture Model (GMM)":
        n_components = st.sidebar.slider("Number of Components", min_value=2, max_value=10, value=5)
        cov_type = st.sidebar.selectbox("Covariance Type", ["full", "tied", "diag", "spherical"])
        
        model_obj, labels, metrics = train_gmm(X_scaled, n_components=n_components, covariance_type=cov_type)
        
        # Diagnostic: AIC/BIC
        gmm_eval_df = run_gmm_evaluation(X_scaled, k_range=range(2, 11), covariance_type=cov_type)
        fig_gmm = px.line(
            gmm_eval_df, 
            x="k", 
            y=["aic", "bic"],
            markers=True,
            title="GMM Diagnostic: AIC & BIC Curves (Lower is Better)",
            labels={"value": "Information Criterion", "k": "Components (k)"}
        )
        fig_gmm.add_vline(x=n_components, line_dash="dash", line_color="#ef4444", annotation_text=f"Selected k={n_components}")
        fig_gmm.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", y=1.1, x=0.5))
        diag_fig = fig_gmm

    elif algorithm_choice == "Mean Shift":
        quantile_val = st.sidebar.slider("Bandwidth Estimation Quantile", min_value=0.05, max_value=0.50, value=0.25, step=0.05)
        model_obj, labels, metrics = train_meanshift(X_scaled, quantile=quantile_val)

    elif algorithm_choice == "Spectral Clustering":
        n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=5)
        affinity_val = st.sidebar.selectbox("Affinity Matrix", ["rbf", "nearest_neighbors"])
        model_obj, labels, metrics = train_spectral(X_scaled, n_clusters=n_clusters, affinity=affinity_val)

    # Display Metrics Banner
    st.markdown(f"### 🎯 Results: **{algorithm_choice}** on **{feature_option}**")
    
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric("Discovered Clusters", f"{metrics['n_clusters']}")
    with m_col2:
        st.metric("Noise Points", f"{metrics['n_noise']}")
    with m_col3:
        sil_display = f"{metrics['silhouette']:.4f}" if metrics['silhouette'] is not None else "N/A"
        st.metric("Silhouette Score", sil_display, help="Higher is better (-1 to +1). Measures cluster separation.")
    with m_col4:
        db_display = f"{metrics['davies_bouldin']:.4f}" if metrics['davies_bouldin'] is not None else "N/A"
        st.metric("Davies-Bouldin Index", db_display, help="Lower is better. Measures cluster tightness & similarity.")
    with m_col5:
        ch_display = f"{metrics['calinski_harabasz']:.1f}" if metrics['calinski_harabasz'] is not None else "N/A"
        st.metric("Calinski-Harabasz", ch_display, help="Higher is better. Ratio of between to within dispersion.")

    # Visualizations
    df_clustered_view = df.copy()
    df_clustered_view["Cluster"] = [f"Cluster {l}" if l != -1 else "Noise (-1)" for l in labels]
    
    # Persona mapping
    profiles_current = get_cluster_profiles(df, labels)
    persona_lookup = dict(zip([f"Cluster {i}" for i in range(len(profiles_current))], profiles_current["Persona"]))
    persona_lookup["Noise (-1)"] = "Unassigned Noise"
    df_clustered_view["Persona"] = df_clustered_view["Cluster"].map(persona_lookup).fillna("Other")

    vcol1, vcol2 = st.columns([1.2, 1])
    
    with vcol1:
        # 2D Main Scatter Plot
        fig_cluster_2d = px.scatter(
            df_clustered_view,
            x="annual_income",
            y="spending_score",
            color="Persona",
            symbol="Cluster",
            hover_data=["customer_id", "gender", "age", "annual_income", "spending_score"],
            title="2D Segmentation: Annual Income vs Spending Score with Personas",
            labels={"annual_income": "Annual Income ($k)", "spending_score": "Spending Score (1-100)"},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        # Add centroids if K-Means or Mean Shift in 2D
        if hasattr(model_obj, "cluster_centers_") and len(selected_features) == 2 and current_scaler is not None:
            raw_centers = current_scaler.inverse_transform(model_obj.cluster_centers_)
            fig_cluster_2d.add_trace(
                go.Scatter(
                    x=raw_centers[:, 0],
                    y=raw_centers[:, 1],
                    mode="markers+text",
                    marker=dict(size=16, color="black", symbol="x", line=dict(width=2, color="white")),
                    text=[f"C{i}" for i in range(len(raw_centers))],
                    textposition="top center",
                    name="Centroids"
                )
            )
            
        fig_cluster_2d.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_cluster_2d, use_container_width=True)
        
    with vcol2:
        # 3D Interactive Scatter Plot
        fig_cluster_3d = px.scatter_3d(
            df_clustered_view,
            x="age",
            y="annual_income",
            z="spending_score",
            color="Persona",
            symbol="Cluster",
            hover_data=["customer_id", "gender"],
            title="3D Cluster Space: Age × Income × Spending",
            labels={"age": "Age", "annual_income": "Income ($k)", "spending_score": "Spending Score"},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_cluster_3d.update_layout(height=480, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_cluster_3d, use_container_width=True)

    # Show diagnostic chart if available
    if diag_fig is not None:
        st.plotly_chart(diag_fig, use_container_width=True)


# ==============================================================================
# TAB 3: MULTI-MODEL BENCHMARK LEADERBOARD
# ==============================================================================
with tabs[2]:
    st.subheader("Clustering Algorithm Benchmark & Comparative Leaderboard")
    st.markdown("Automated side-by-side performance evaluation across all 6 clustering techniques on the standard 2D feature set (*Annual Income* & *Spending Score*).")
    
    # Run full comparison
    X_std, _ = get_feature_matrix(df, ["annual_income", "spending_score"], scaler_type="standard")
    
    benchmarks = []
    
    # 1. K-Means
    _, km_lab, km_m = train_kmeans(X_std, n_clusters=5)
    benchmarks.append({"Algorithm": "K-Means (k=5)", "Type": "Centroid-Based", **km_m})
    
    # 2. Hierarchical Ward
    _, hw_lab, hw_m = train_hierarchical(X_std, n_clusters=5, linkage_method="ward")
    benchmarks.append({"Algorithm": "Hierarchical (Ward Linkage, k=5)", "Type": "Agglomerative Tree", **hw_m})
    
    # 3. Hierarchical Complete
    _, hc_lab, hc_m = train_hierarchical(X_std, n_clusters=5, linkage_method="complete")
    benchmarks.append({"Algorithm": "Hierarchical (Complete Linkage, k=5)", "Type": "Agglomerative Tree", **hc_m})
    
    # 4. GMM
    _, gmm_lab, gmm_m = train_gmm(X_std, n_components=5)
    benchmarks.append({"Algorithm": "Gaussian Mixture Model (GMM, k=5)", "Type": "Probabilistic / Density", **gmm_m})
    
    # 5. DBSCAN
    _, db_lab, db_m = train_dbscan(X_std, eps=0.45, min_samples=4)
    benchmarks.append({"Algorithm": "DBSCAN (eps=0.45, min_samples=4)", "Type": "Density-Based", **db_m})
    
    # 6. Mean Shift
    _, ms_lab, ms_m = train_meanshift(X_std, quantile=0.25)
    benchmarks.append({"Algorithm": "Mean Shift", "Type": "Centroid Non-Parametric", **ms_m})
    
    # 7. Spectral
    try:
        _, sp_lab, sp_m = train_spectral(X_std, n_clusters=5)
        benchmarks.append({"Algorithm": "Spectral Clustering (k=5)", "Type": "Graph / Spectral Graph", **sp_m})
    except Exception:
        pass

    bench_df = pd.DataFrame(benchmarks)
    
    # Leaderboard Table
    b_col1, b_col2 = st.columns([1.1, 0.9])
    
    with b_col1:
        st.markdown("#### 🏆 Leaderboard Summary")
        st.dataframe(
            bench_df.sort_values(by="silhouette", ascending=False).style.format({
                "silhouette": "{:.4f}",
                "davies_bouldin": "{:.4f}",
                "calinski_harabasz": "{:.2f}"
            }).background_gradient(subset=["silhouette"], cmap="Greens")
            .background_gradient(subset=["davies_bouldin"], cmap="Blues_r"),
            use_container_width=True,
            height=300
        )
        
    with b_col2:
        # Comparison Bar Chart
        fig_bench_bar = px.bar(
            bench_df.sort_values(by="silhouette", ascending=True),
            y="Algorithm",
            x="silhouette",
            color="silhouette",
            orientation="h",
            color_continuous_scale="Viridis",
            title="Silhouette Score by Algorithm (Higher is Better)",
            labels={"silhouette": "Silhouette Score", "Algorithm": ""}
        )
        fig_bench_bar.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_bench_bar, use_container_width=True)

    # Benchmark Insights Callout
    st.markdown("""
    > [!TIP]
    > **Key Benchmark Insights**:
    > - **K-Means (k=5)** and **Hierarchical Clustering (Ward)** achieve the highest Silhouette Score (~0.5546), forming crisp, compact, well-separated convex clusters.
    > - **Gaussian Mixture Models (GMM)** provide soft probabilistic memberships, ideal when customers transition between spending habits.
    > - **DBSCAN** excels at identifying density anomalies and noise points without forcing edge customers into unnatural clusters.
    """)


# ==============================================================================
# TAB 4: BUSINESS PERSONAS & MARKETING STRATEGIES
# ==============================================================================
with tabs[3]:
    st.subheader("Customer Personas & Actionable Marketing Strategies")
    st.markdown("Detailed customer archetypes derived from K-Means (k=5) clustering on the Mall Customer dataset.")
    
    # Calculate profiles
    X_std, _ = get_feature_matrix(df, ["annual_income", "spending_score"], scaler_type="standard")
    _, km_labels_prod, _ = train_kmeans(X_std, n_clusters=5)
    profiles = get_cluster_profiles(df, km_labels_prod)
    
    # Radar Chart of Cluster Personas
    categories = ["Avg Age (norm)", "Avg Income (norm)", "Avg Spending (norm)", "Female Share (%)"]
    
    fig_radar = go.Figure()
    
    # Normalize for radar comparison
    for _, row in profiles.iterrows():
        norm_age = (row["Avg Age"] - df["age"].min()) / (df["age"].max() - df["age"].min()) * 100
        norm_inc = (row["Avg Income ($k)"] - df["annual_income"].min()) / (df["annual_income"].max() - df["annual_income"].min()) * 100
        norm_spn = row["Avg Spending Score"]  # already 0-100
        fem_share = row["Female (%)"]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=[norm_age, norm_inc, norm_spn, fem_share, norm_age],
            theta=categories + [categories[0]],
            fill="toself",
            name=f"{row['Cluster ID']}: {row['Persona']}",
            line=dict(color=row["Badge Color"])
        ))
        
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Persona Attribute Radar Comparison",
        height=380,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    rcol1, rcol2 = st.columns([1, 1.2])
    with rcol1:
        st.plotly_chart(fig_radar, use_container_width=True)
    with rcol2:
        # Cluster Size Donut Chart
        fig_donut = px.pie(
            profiles,
            values="Count",
            names="Persona",
            color="Persona",
            color_discrete_sequence=profiles["Badge Color"].tolist(),
            hole=0.45,
            title="Customer Base Distribution by Segment"
        )
        fig_donut.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_donut, use_container_width=True)

    # Render Persona Cards
    st.markdown("### 📋 Persona Playbooks")
    
    for _, row in profiles.iterrows():
        st.markdown(f"""
        <div class="persona-card" style="border-left: 6px solid {row['Badge Color']};">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 12px;">
                <div>
                    <span class="badge" style="background-color: {row['Badge Color']}20; color: {row['Badge Color']};">
                        {row['Cluster ID']}
                    </span>
                    <h3 style="display: inline; margin-left: 10px; color: #0f172a;">{row['Persona']}</h3>
                </div>
                <div style="font-weight: 700; color: #475569; font-size: 0.95rem;">
                    {row['Count']} Customers ({row['Share (%)']}% of base)
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; background: #f8fafc; padding: 12px 16px; border-radius: 10px; margin-bottom: 12px; font-size: 0.9rem;">
                <div><strong>Avg Age:</strong> {row['Avg Age']} yrs</div>
                <div><strong>Avg Income:</strong> ${row['Avg Income ($k)']}k / yr</div>
                <div><strong>Spending Score:</strong> {row['Avg Spending Score']} / 100</div>
                <div><strong>Gender:</strong> {row['Female (%)']}% ♀, {row['Male (%)']}% ♂</div>
            </div>
            <p style="color: #334155; margin-bottom: 8px;"><strong>Archetype Profile:</strong> {row['Description']}</p>
            <p style="color: #1e293b; margin-bottom: 8px;"><strong>🎯 Recommended Strategy:</strong> {row['Marketing Strategy']}</p>
            <p style="color: #6366f1; margin-bottom: 0;"><strong>📣 Best Marketing Channels:</strong> {row['Recommended Channel']}</p>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# TAB 5: LIVE CUSTOMER SEGMENT PREDICTOR
# ==============================================================================
with tabs[4]:
    st.subheader("🎯 Real-Time Customer Segment Predictor")
    st.markdown("Enter a customer's demographic and financial parameters to dynamically assign them to an optimal cluster segment and retrieve tailored marketing recommendations.")
    
    # Train the standard production model
    X_std, prod_scaler = get_feature_matrix(df, ["annual_income", "spending_score"], scaler_type="standard")
    prod_model, prod_labels, _ = train_kmeans(X_std, n_clusters=5)
    prod_profiles = get_cluster_profiles(df, prod_labels)
    
    col_input, col_pred = st.columns([1, 1.2])
    
    with col_input:
        st.markdown("#### 📝 Customer Profile Inputs")
        
        with st.form("customer_input_form"):
            in_gender = st.selectbox("Customer Gender", ["Female", "Male"])
            in_age = st.slider("Customer Age", min_value=18, max_value=75, value=32, step=1)
            in_income = st.slider("Annual Income ($k)", min_value=10, max_value=140, value=75, step=1)
            in_spending = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=82, step=1)
            
            submitted = st.form_submit_button("🚀 Segment Customer & Get Strategy", use_container_width=True)

    with col_pred:
        customer_dict = {
            "gender": in_gender,
            "age": in_age,
            "annual_income": in_income,
            "spending_score": in_spending
        }
        
        pred_result = predict_new_customer(
            customer_data=customer_dict,
            model=prod_model,
            scaler=prod_scaler,
            feature_cols=["annual_income", "spending_score"]
        )
        
        assigned_c = pred_result["assigned_cluster"]
        persona_info = pred_result["persona_info"]
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {persona_info['badge_color']}15 0%, #ffffff 100%); border: 2px solid {persona_info['badge_color']}; border-radius: 16px; padding: 24px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.05);">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span class="badge" style="background-color: {persona_info['badge_color']}; color: white;">
                    Assigned Cluster {assigned_c}
                </span>
                <span style="font-weight: 600; color: #64748b; font-size: 0.85rem;">{persona_info['age_tag']}</span>
            </div>
            <h2 style="color: #0f172a; margin: 12px 0 6px 0; font-weight: 800;">
                {persona_info['persona_name']}
            </h2>
            <p style="color: #475569; font-size: 0.95rem; margin-bottom: 16px;">
                {persona_info['description']}
            </p>
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px; margin-bottom: 14px;">
                <div style="font-weight: 700; color: #0f172a; margin-bottom: 4px;">🎯 Recommended Marketing Play:</div>
                <div style="color: #334155; font-size: 0.9rem;">{persona_info['marketing_action']}</div>
            </div>
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px;">
                <div style="font-weight: 700; color: #0f172a; margin-bottom: 4px;">📣 Optimal Channels:</div>
                <div style="color: #6366f1; font-size: 0.9rem;">{persona_info['recommended_channel']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # Plot customer on the 2D cluster map
    st.markdown("#### 📍 Customer Position in Segmentation Space")
    
    df_plot_pred = df.copy()
    df_plot_pred["Cluster"] = [f"Cluster {l}" for l in prod_labels]
    
    fig_pred_map = px.scatter(
        df_plot_pred,
        x="annual_income",
        y="spending_score",
        color="Cluster",
        hover_data=["customer_id", "age", "gender"],
        title="Visual Customer Mapping vs Cluster Landscape",
        labels={"annual_income": "Annual Income ($k)", "spending_score": "Spending Score (1-100)"},
        color_discrete_sequence=px.colors.qualitative.Bold,
        opacity=0.65
    )
    
    # Add input customer beacon
    fig_pred_map.add_trace(
        go.Scatter(
            x=[in_income],
            y=[in_spending],
            mode="markers+text",
            marker=dict(size=22, color=persona_info["badge_color"], symbol="star", line=dict(width=3, color="black")),
            text=["📍 NEW CUSTOMER"],
            textposition="top center",
            name="New Customer"
        )
    )
    
    fig_pred_map.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_pred_map, use_container_width=True)


# ==============================================================================
# TAB 6: DATA EXPLORER & EXPORT
# ==============================================================================
with tabs[5]:
    st.subheader("Data Explorer & Export Segmented Records")
    
    # Generate full segmented table
    X_std, _ = get_feature_matrix(df, ["annual_income", "spending_score"], scaler_type="standard")
    _, full_labels, _ = train_kmeans(X_std, n_clusters=5)
    
    full_profiles = get_cluster_profiles(df, full_labels)
    persona_map = dict(zip(range(len(full_profiles)), full_profiles["Persona"]))
    strategy_map = dict(zip(range(len(full_profiles)), full_profiles["Marketing Strategy"]))
    
    df_export = df.copy()
    df_export["Cluster_ID"] = full_labels
    df_export["Persona"] = df_export["Cluster_ID"].map(persona_map)
    df_export["Targeted_Strategy"] = df_export["Cluster_ID"].map(strategy_map)
    
    # Filters
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        sel_gender = st.multiselect("Filter by Gender", options=["Female", "Male"], default=["Female", "Male"])
    with fcol2:
        sel_persona = st.multiselect("Filter by Persona", options=df_export["Persona"].unique(), default=df_export["Persona"].unique())
    with fcol3:
        income_filter = st.slider("Filter by Annual Income Range ($k)", int(df["annual_income"].min()), int(df["annual_income"].max()), (int(df["annual_income"].min()), int(df["annual_income"].max())))
        
    df_filtered = df_export[
        (df_export["gender"].isin(sel_gender)) &
        (df_export["Persona"].isin(sel_persona)) &
        (df_export["annual_income"].between(income_filter[0], income_filter[1]))
    ]
    
    st.dataframe(df_filtered, use_container_width=True, height=350)
    
    # Download button
    csv_data = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Segmented Customer Data (CSV)",
        data=csv_data,
        file_name="mall_customers_segmented.csv",
        mime="text/csv",
        use_container_width=True
    )
