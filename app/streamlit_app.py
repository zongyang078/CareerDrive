"""
CareerDrive Streamlit Demo

Three pages:
1. Industry Dashboard — descriptive stats and landscape overview
2. Clustering Explorer — interactive PCA scatter plot with cluster details
3. Career Match — input skills, get matched occupations

Run with: streamlit run app/streamlit_app.py
"""

import sys
sys.path.append('..')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# --- Page Config ---
st.set_page_config(page_title="CareerDrive", page_icon="\U0001f6e4\ufe0f", layout="wide")

# --- Load Data ---
@st.cache_data
def load_data():
    features = pd.read_csv('data/processed/occupation_features.csv', index_col=0)
    clusters = pd.read_csv('data/processed/cluster_labels.csv')
    job_zones = pd.read_csv('data/processed/job_zones.csv')
    related = pd.read_csv('data/processed/related_occupations.csv')
    agc = pd.read_csv('data/processed/agc_members.csv')
    apprentice = pd.read_csv('data/processed/apprenticeships.csv')
    cc = pd.read_csv('data/processed/community_college.csv')
    umaine = pd.read_csv('data/processed/umaine_programs.csv')

    titles = features['Title']
    X_raw = features.drop(columns='Title')
    scaler = StandardScaler().fit(X_raw)
    X_scaled = scaler.transform(X_raw)

    pca = PCA(n_components=2).fit(X_scaled)
    pca2 = pca.transform(X_scaled)

    return {
        'features': features, 'clusters': clusters, 'job_zones': job_zones,
        'related': related, 'agc': agc, 'apprentice': apprentice,
        'cc': cc, 'umaine': umaine, 'titles': titles,
        'X_raw': X_raw, 'scaler': scaler, 'X_scaled': X_scaled,
        'pca': pca, 'pca2': pca2,
    }

data = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("\U0001f6e4\ufe0f CareerDrive")
page = st.sidebar.radio("Navigate", ["Industry Dashboard", "Clustering Explorer", "Career Match"])

# ================================================================
# PAGE 1: INDUSTRY DASHBOARD
# ================================================================
if page == "Industry Dashboard":
    st.title("Maine Construction Industry Overview")
    st.markdown("A snapshot of the workforce ecosystem: companies, occupations, and training programs.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Companies", len(data['agc']))
    col2.metric("Occupations", len(data['clusters']))
    col3.metric("Apprenticeships", len(data['apprentice']))
    col4.metric("Education Programs", len(data['cc']) + len(data['umaine']))

    st.divider()

    # Company distribution
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Companies by Type")
        company_counts = data['agc']['type'].value_counts().reset_index()
        company_counts.columns = ['Type', 'Count']
        fig = px.bar(company_counts, x='Type', y='Count', color='Type',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Occupations by Cluster")
        cluster_counts = data['clusters']['cluster_name'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        fig = px.pie(cluster_counts, values='Count', names='Cluster',
                     color_discrete_sequence=['#2E86AB', '#E8593C', '#1B998B'])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Job Zone distribution
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Occupations by Job Zone (Complexity)")
        jz_counts = data['clusters'].groupby('Job Zone').size().reset_index(name='Count')
        jz_counts['Job Zone'] = jz_counts['Job Zone'].astype(str)
        fig = px.bar(jz_counts, x='Job Zone', y='Count',
                     color='Job Zone', color_discrete_sequence=px.colors.sequential.Teal)
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Apprenticeship Training Hours")
        app_data = data['apprentice'].sort_values('term_hours', ascending=True)
        fig = px.bar(app_data, x='term_hours', y='title', orientation='h',
                     color='term_hours', color_continuous_scale='Teal')
        fig.update_layout(height=400, yaxis_title='', xaxis_title='Hours',
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Education programs
    st.subheader("Education Programs")
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**Community College Programs**")
        st.dataframe(data['cc'][['college', 'program_name', 'credentials']],
                     hide_index=True, use_container_width=True)
    with e2:
        st.markdown("**UMaine Programs**")
        st.dataframe(data['umaine'][['campus', 'program_name', 'degree_type']],
                     hide_index=True, use_container_width=True)


# ================================================================
# PAGE 2: CLUSTERING EXPLORER
# ================================================================
elif page == "Clustering Explorer":
    st.title("Occupation Clustering Analysis")
    st.markdown("20 construction occupations clustered by 120 skill/knowledge/ability features using PCA + K-Means.")

    # PCA scatter plot
    pca2 = data['pca2']
    clusters = data['clusters']
    pca_obj = data['pca']

    plot_df = pd.DataFrame({
        'PC1': pca2[:, 0],
        'PC2': pca2[:, 1],
        'Title': clusters['Title'],
        'Cluster': clusters['cluster_name'],
        'Job Zone': clusters['Job Zone'].astype(str),
        'Code': clusters['O*NET-SOC Code'],
    })

    color_map = {
        'Management/Engineering': '#2E86AB',
        'Skilled Trades': '#1B998B',
        'Entry Level/Operators': '#E8593C',
    }

    fig = px.scatter(plot_df, x='PC1', y='PC2', color='Cluster',
                     hover_data=['Title', 'Job Zone', 'Code'],
                     text='Title', color_discrete_map=color_map,
                     width=800, height=550)
    fig.update_traces(textposition='top center', textfont_size=10, marker_size=12)
    fig.update_layout(
        xaxis_title=f"PC1 ({pca_obj.explained_variance_ratio_[0]:.1%} variance)",
        yaxis_title=f"PC2 ({pca_obj.explained_variance_ratio_[1]:.1%} variance)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Cluster detail
    st.subheader("Cluster Details")
    selected_cluster = st.selectbox("Select a cluster:", list(color_map.keys()))
    cluster_members = clusters[clusters['cluster_name'] == selected_cluster]

    st.dataframe(
        cluster_members[['O*NET-SOC Code', 'Title', 'Job Zone']],
        hide_index=True, use_container_width=True
    )

    # Skill profile radar chart for selected cluster
    st.subheader(f"Average Skill Profile: {selected_cluster}")
    member_codes = cluster_members['O*NET-SOC Code'].tolist()
    X_raw = data['X_raw']

    # Get top 10 most distinguishing features for this cluster
    cluster_mean = X_raw.loc[member_codes].mean()
    overall_mean = X_raw.mean()
    diff = (cluster_mean - overall_mean).sort_values(ascending=False)
    top_features = diff.head(10).index.tolist()

    radar_data = pd.DataFrame({
        'Feature': [f.split('_', 1)[1] for f in top_features],
        selected_cluster: cluster_mean[top_features].values,
        'Overall Average': overall_mean[top_features].values,
    })

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=radar_data[selected_cluster], theta=radar_data['Feature'],
        fill='toself', name=selected_cluster, fillcolor='rgba(46,134,171,0.3)',
        line_color='#2E86AB',
    ))
    fig.add_trace(go.Scatterpolar(
        r=radar_data['Overall Average'], theta=radar_data['Feature'],
        fill='toself', name='Overall Average', fillcolor='rgba(200,200,200,0.2)',
        line_color='#999999',
    ))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 5])), height=400)
    st.plotly_chart(fig, use_container_width=True)


# ================================================================
# PAGE 3: CAREER MATCH
# ================================================================
elif page == "Career Match":
    st.title("Career Match Tool")
    st.markdown("Select your skills and experience level to find matching construction careers.")

    # Group features by category for cleaner UI
    X_raw = data['X_raw']
    skill_cols = [c for c in X_raw.columns if c.startswith('skill_')]
    knowledge_cols = [c for c in X_raw.columns if c.startswith('knowledge_')]

    # Let user pick top skills
    st.subheader("1. Select your top skills")
    skill_names = [c.replace('skill_', '') for c in skill_cols]
    selected_skills = st.multiselect("Choose skills you're strong at (pick 3-7):",
                                      skill_names, default=['Critical Thinking'])

    st.subheader("2. Select your knowledge areas")
    knowledge_names = [c.replace('knowledge_', '') for c in knowledge_cols]
    selected_knowledge = st.multiselect("Choose knowledge areas (pick 2-5):",
                                         knowledge_names, default=['Building and Construction'])

    st.subheader("3. Rate your experience level")
    exp_level = st.slider("Overall experience level", 1.0, 5.0, 3.0, 0.5,
                          help="1 = Beginner, 3 = Intermediate, 5 = Expert")

    if st.button("Find Matching Careers", type="primary"):
        # Build user vector
        user_skills = {}
        for s in selected_skills:
            user_skills[f'skill_{s}'] = exp_level
        for k in selected_knowledge:
            user_skills[f'knowledge_{k}'] = exp_level

        if len(user_skills) < 2:
            st.warning("Please select at least 2 skills or knowledge areas.")
        else:
            # Compute similarity on selected dimensions only
            valid_features = [f for f in user_skills.keys() if f in X_raw.columns]
            X_sub = X_raw[valid_features]
            user_vec = pd.Series(user_skills)[valid_features].values.reshape(1, -1)

            sub_scaler = StandardScaler().fit(X_sub)
            X_sub_scaled = sub_scaler.transform(X_sub)
            user_sub_scaled = sub_scaler.transform(user_vec)

            sims = cosine_similarity(user_sub_scaled, X_sub_scaled)[0]

            results = pd.DataFrame({
                'Occupation': data['titles'].values,
                'Cluster': data['clusters']['cluster_name'].values,
                'Job Zone': data['clusters']['Job Zone'].values,
                'Match Score': np.round(sims * 100, 1),
            }, index=data['features'].index).sort_values('Match Score', ascending=False)

            st.divider()
            st.subheader("Your Top Matches")

            # Top 5 results
            top5 = results.head(5)
            for i, (_, row) in enumerate(top5.iterrows()):
                col1, col2, col3 = st.columns([3, 2, 1])
                col1.markdown(f"**{i+1}. {row['Occupation']}**")
                col2.markdown(f"`{row['Cluster']}` | Job Zone {row['Job Zone']}")
                col3.metric("Match", f"{row['Match Score']}%")

            st.divider()

            # Full ranking table
            with st.expander("View all occupations ranked"):
                st.dataframe(results, hide_index=True, use_container_width=True)