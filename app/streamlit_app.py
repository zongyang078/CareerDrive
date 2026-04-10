"""
CareerDrive Streamlit Demo — Refactored

Three pages:
1. Industry Dashboard — descriptive stats, company landscape, supply-demand gap
2. Clustering Explorer — interactive PCA scatter plot with cluster details
3. Career Match — input skills, get matched occupations + training pathways

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

# --- Consistent Color Palette ---
CLUSTER_COLORS = {
    'Management/Engineering': '#2E86AB',
    'Skilled Trades': '#1B998B',
    'Entry Level/Operators': '#E8593C',
}

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
    companies = pd.read_csv('data/processed/companies_merged.csv')

    titles = features['Title']
    X_raw = features.drop(columns='Title')
    scaler = StandardScaler().fit(X_raw)
    X_scaled = scaler.transform(X_raw)

    pca = PCA(n_components=2).fit(X_scaled)
    pca2 = pca.transform(X_scaled)

    # Manual mappings for gap analysis
    apprentice_to_onet = {
        'Arborist': None,
        'Bridge Carpenter/Heavy Highway': '47-2031.00',
        'Construction Carpenter': '47-2031.00',
        'Construction Craft Concrete Laborer': None,
        'Construction Craft Heavy / Highway Laborer': None,
        'Construction Equipment Operator': '47-2073.00',
        'Construction Specialist': None,
        'Crane Mechanic': None,
        'Crane Operator': '47-2073.00',
        'Earthworks Laborer': None,
        'Electrician': '47-2111.00',
        'Fencing Installer': None,
        'Firestopping Installer': None,
        'Firestopping Technician': None,
        'Foreman': '47-1011.00',
        'Lead Logging Equipment Operator': None,
        'Marine Carpenter - Heavy Civil': '47-2031.00',
        'Solar Mechanical Installation Technician': None,
        'Welder': None,
    }
    apprentice['onet_match'] = apprentice['title'].map(apprentice_to_onet)

    cc_to_cluster = {
        'Building Construction Technology': 'Skilled Trades',
        'HVAC/R Technology': 'Skilled Trades',
        'Plumbing & Heating Technology': 'Skilled Trades',
        'Building Construction': 'Skilled Trades',
        'Electrical Technology': 'Skilled Trades',
        'Precision Machining Technology': 'Skilled Trades',
        'Welding Technology': 'Skilled Trades',
        'Heating Technology': 'Skilled Trades',
        'Plumbing Technology': 'Skilled Trades',
        'Electrical Lineworker Technology': 'Skilled Trades',
        'Heavy Equipment Operations': 'Entry Level/Operators',
        'Civil Engineering Technology': 'Management/Engineering',
        'Engineering Technology': 'Management/Engineering',
        'Architectural & Civil Engineering Technology': 'Management/Engineering',
        'Land Surveying Technology': 'Management/Engineering',
    }
    cc['cluster_match'] = cc['program_name'].map(cc_to_cluster).fillna('Unmatched')

    umaine_to_cluster = {
        'Construction Engineering Technology': 'Management/Engineering',
        'Architecture (5-Year Professional Degree)': 'Management/Engineering',
        'Civil & Environmental Engineering': 'Management/Engineering',
        'Mechanical Engineering': 'Management/Engineering',
        'Electrical Engineering Technology': 'Skilled Trades',
        'Surveying Engineering Technology': 'Management/Engineering',
        'Facilities Management': 'Skilled Trades',
        'Construction Management Technology': 'Management/Engineering',
    }
    umaine['cluster_match'] = umaine['program_name'].map(umaine_to_cluster).fillna('Unmatched')

    return {
        'features': features, 'clusters': clusters, 'job_zones': job_zones,
        'related': related, 'agc': agc, 'apprentice': apprentice,
        'cc': cc, 'umaine': umaine, 'companies': companies,
        'titles': titles, 'X_raw': X_raw, 'scaler': scaler,
        'X_scaled': X_scaled, 'pca': pca, 'pca2': pca2,
    }

data = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("\U0001f6e4\ufe0f CareerDrive")
st.sidebar.markdown("*Maine Construction Industry Career Map*")
page = st.sidebar.radio("Navigate", ["Industry Dashboard", "Clustering Explorer", "Career Match"])
st.sidebar.divider()
st.sidebar.markdown("DS 5230 — Unsupervised ML")
st.sidebar.markdown("Spring 2026 Final Project")

# ================================================================
# PAGE 1: INDUSTRY DASHBOARD
# ================================================================
if page == "Industry Dashboard":
    st.title("\U0001f3d7\ufe0f Maine Construction Industry Overview")
    st.markdown("A snapshot of the workforce ecosystem: companies, occupations, training programs, and workforce gaps.")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Companies", len(data['companies']))
    col2.metric("Occupations", len(data['clusters']))
    col3.metric("Apprenticeships", len(data['apprentice']))
    col4.metric("Education Programs", len(data['cc']) + len(data['umaine']))

    st.divider()

    # Company distribution
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Companies by Type")
        company_counts = data['companies']['type'].value_counts().reset_index()
        company_counts.columns = ['Type', 'Count']
        fig = px.bar(company_counts, x='Type', y='Count', color='Type',
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     text='Count')
        fig.update_layout(showlegend=False, height=350)
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Occupations by Cluster")
        cluster_counts = data['clusters']['cluster_name'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        fig = px.pie(cluster_counts, values='Count', names='Cluster',
                     color='Cluster', color_discrete_map=CLUSTER_COLORS)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Job Zone and apprenticeship info
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Occupations by Job Zone (Complexity)")
        jz_data = data['clusters'].copy()
        jz_counts = jz_data.groupby('Job Zone').size().reset_index(name='Count')
        jz_counts['Job Zone'] = jz_counts['Job Zone'].astype(str)
        fig = px.bar(jz_counts, x='Job Zone', y='Count',
                     color='Job Zone', color_discrete_sequence=px.colors.sequential.Teal,
                     text='Count')
        fig.update_layout(showlegend=False, height=300)
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Cluster \u00d7 Job Zone Distribution")
        ct = pd.crosstab(data['clusters']['cluster_name'], data['clusters']['Job Zone'])
        fig = px.imshow(ct, text_auto=True, color_continuous_scale='YlOrRd',
                        labels=dict(x='Job Zone', y='Cluster', color='Count'))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Supply-Demand Gap Analysis
    st.subheader("\U0001f4ca Supply-Demand Gap Analysis")
    st.markdown("""
    The **demand** side represents the 20 key occupations across 3 clusters.
    The **supply** side represents available training pathways (apprenticeships + education programs).
    """)

    cluster_map = dict(zip(data['clusters']['O*NET-SOC Code'], data['clusters']['cluster_name']))
    codes_with_app = data['apprentice'].dropna(subset=['onet_match'])['onet_match'].unique()
    all_programs = pd.concat([
        data['cc'][data['cc']['cluster_match'] != 'Unmatched'][['program_name', 'cluster_match']].rename(
            columns={'program_name': 'program', 'cluster_match': 'cluster'}),
        data['umaine'][data['umaine']['cluster_match'] != 'Unmatched'][['program_name', 'cluster_match']].rename(
            columns={'program_name': 'program', 'cluster_match': 'cluster'}),
    ])

    gap_rows = []
    for c_name in ['Management/Engineering', 'Skilled Trades', 'Entry Level/Operators']:
        cluster_codes = data['clusters'][data['clusters']['cluster_name'] == c_name]['O*NET-SOC Code']
        n_occ = len(cluster_codes)
        n_app = sum(1 for c in cluster_codes if c in codes_with_app)
        n_edu = len(all_programs[all_programs['cluster'] == c_name])
        gap_rows.append({
            'Cluster': c_name, 'Occupations': n_occ,
            'Apprenticeships': n_app, 'Education Programs': n_edu,
            'Total Supply': n_app + n_edu,
            'Coverage Ratio': round((n_app + n_edu) / n_occ, 2) if n_occ else 0
        })
    gap_df = pd.DataFrame(gap_rows)

    g1, g2 = st.columns(2)
    with g1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Occupations (Demand)', x=gap_df['Cluster'],
                             y=gap_df['Occupations'], marker_color='#2E86AB'))
        fig.add_trace(go.Bar(name='Apprenticeships', x=gap_df['Cluster'],
                             y=gap_df['Apprenticeships'], marker_color='#1B998B'))
        fig.add_trace(go.Bar(name='Education Programs', x=gap_df['Cluster'],
                             y=gap_df['Education Programs'], marker_color='#4CAF50'))
        fig.update_layout(barmode='group', height=350, title='Supply vs Demand by Cluster')
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        colors = ['#E8593C' if r < 1.0 else '#4CAF50' for r in gap_df['Coverage Ratio']]
        fig = go.Figure(go.Bar(x=gap_df['Cluster'], y=gap_df['Coverage Ratio'],
                               marker_color=colors, text=gap_df['Coverage Ratio']))
        fig.add_hline(y=1.0, line_dash='dash', annotation_text='1:1 Ratio')
        fig.update_layout(height=350, title='Training Coverage Ratio',
                          yaxis_title='Supply / Demand')
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("View detailed gap table"):
        st.dataframe(gap_df, hide_index=True, use_container_width=True)

    st.divider()

    # Education programs table
    st.subheader("Education Programs")
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**Community College Programs**")
        st.dataframe(data['cc'][['college', 'program_name', 'credentials', 'cluster_match']],
                     hide_index=True, use_container_width=True)
    with e2:
        st.markdown("**UMaine Programs**")
        st.dataframe(data['umaine'][['campus', 'program_name', 'degree_type', 'cluster_match']],
                     hide_index=True, use_container_width=True)


# ================================================================
# PAGE 2: CLUSTERING EXPLORER
# ================================================================
elif page == "Clustering Explorer":
    st.title("\U0001f50d Occupation Clustering Analysis")
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

    fig = px.scatter(plot_df, x='PC1', y='PC2', color='Cluster',
                     hover_data=['Title', 'Job Zone', 'Code'],
                     text='Title', color_discrete_map=CLUSTER_COLORS,
                     width=900, height=600)
    fig.update_traces(textposition='top center', textfont_size=9, marker_size=12)
    fig.update_layout(
        xaxis_title=f"PC1 ({pca_obj.explained_variance_ratio_[0]:.1%} variance)",
        yaxis_title=f"PC2 ({pca_obj.explained_variance_ratio_[1]:.1%} variance)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Cluster detail
    st.subheader("Cluster Details")
    selected_cluster = st.selectbox("Select a cluster:", list(CLUSTER_COLORS.keys()))
    cluster_members = clusters[clusters['cluster_name'] == selected_cluster]

    st.dataframe(
        cluster_members[['O*NET-SOC Code', 'Title', 'Job Zone']],
        hide_index=True, use_container_width=True
    )

    # Skill profile radar chart
    st.subheader(f"Average Skill Profile: {selected_cluster}")
    member_codes = cluster_members['O*NET-SOC Code'].tolist()
    X_raw = data['X_raw']

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
        fill='toself', name=selected_cluster,
        fillcolor=f"rgba({','.join(str(int(CLUSTER_COLORS[selected_cluster][i:i+2], 16)) for i in (1,3,5))},0.3)",
        line_color=CLUSTER_COLORS[selected_cluster],
    ))
    fig.add_trace(go.Scatterpolar(
        r=radar_data['Overall Average'], theta=radar_data['Feature'],
        fill='toself', name='Overall Average', fillcolor='rgba(200,200,200,0.2)',
        line_color='#999999',
    ))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 5])), height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # === Career Path Network Graph ===
    st.subheader("🗺️ Career Path Network")
    st.markdown("Nodes are occupations, edges from O\\*NET Related Occupations + cosine similarity (≥0.95). "
                "Node size reflects Job Zone complexity.")

    import networkx as nx

    related = data['related']
    features_df = data['features']
    our_codes = features_df.index.tolist()
    title_lookup = dict(zip(data['clusters']['O*NET-SOC Code'], data['clusters']['Title']))
    zone_lookup = dict(zip(data['clusters']['O*NET-SOC Code'], data['clusters']['Job Zone']))
    cluster_lookup = dict(zip(data['clusters']['O*NET-SOC Code'], data['clusters']['cluster_name']))

    # Build graph
    G = nx.Graph()
    for code in our_codes:
        G.add_node(code)

    for _, row in related.iterrows():
        src, dst = row['O*NET-SOC Code'], row['Related O*NET-SOC Code']
        if src in our_codes and dst in our_codes:
            G.add_edge(src, dst, source='onet')

    # Add cosine similarity edges
    cos_sim = pd.read_csv('data/processed/cosine_similarity.csv', index_col=0)
    for i, c1 in enumerate(our_codes):
        for j, c2 in enumerate(our_codes):
            if i >= j:
                continue
            if c1 in cos_sim.index and c2 in cos_sim.columns:
                if cos_sim.loc[c1, c2] >= 0.95 and not G.has_edge(c1, c2):
                    G.add_edge(c1, c2, source='similarity')

    # Structured layout: X = cluster column, Y = Job Zone row
    cluster_x = {'Entry Level/Operators': 0, 'Skilled Trades': 1, 'Management/Engineering': 2}

    # Within each cluster × job zone cell, spread nodes horizontally
    from collections import defaultdict
    cell_counts = defaultdict(int)
    pos = {}
    for code in our_codes:
        cname = cluster_lookup.get(code, 'Skilled Trades')
        jz = zone_lookup.get(code, 2)
        cx = cluster_x[cname]
        cell_key = (cx, jz)
        offset = cell_counts[cell_key] * 0.15
        cell_counts[cell_key] += 1
        pos[code] = (cx + offset - 0.15, jz)

    # Build Plotly edge traces
    edge_traces = []
    for src, dst, edata in G.edges(data=True):
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        color = 'rgba(100,100,100,0.4)' if edata.get('source') == 'onet' else 'rgba(100,100,100,0.15)'
        width = 1.5 if edata.get('source') == 'onet' else 0.5
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines', line=dict(width=width, color=color),
            hoverinfo='none', showlegend=False,
        ))

    # One trace per cluster for legend
    node_traces = []
    for cname, color in CLUSTER_COLORS.items():
        codes_in = [c for c in our_codes if cluster_lookup.get(c) == cname]
        node_traces.append(go.Scatter(
            x=[pos[c][0] for c in codes_in],
            y=[pos[c][1] for c in codes_in],
            mode='markers+text',
            marker=dict(
                size=[zone_lookup.get(c, 3) * 10 + 10 for c in codes_in],
                color=color, line=dict(width=2, color='white'),
                opacity=0.9,
            ),
            text=[title_lookup.get(c, c).split(',')[0][:30] for c in codes_in],
            textposition='top center',
            textfont=dict(size=9),
            hovertext=[f"<b>{title_lookup.get(c, c)}</b><br>Job Zone: {zone_lookup.get(c, '')}<br>"
                        f"Cluster: {cname}<br>Connections: {G.degree(c)}" for c in codes_in],
            hoverinfo='text',
            name=cname,
        ))

    fig = go.Figure(data=edge_traces + node_traces)
    fig.update_layout(
        height=700,
        showlegend=True,
        xaxis=dict(
            showgrid=False, zeroline=False,
            tickvals=[0, 1, 2],
            ticktext=['Entry Level /<br>Operators', 'Skilled<br>Trades', 'Management /<br>Engineering'],
            tickfont=dict(size=12, color='#333'),
            side='bottom',
        ),
        yaxis=dict(
            showgrid=True, gridcolor='rgba(200,200,200,0.3)',
            zeroline=False,
            tickvals=[2, 3, 4, 5],
            ticktext=['Zone 2<br>(Entry)', 'Zone 3<br>(Mid)', 'Zone 4<br>(Advanced)', 'Zone 5<br>(Expert)'],
            tickfont=dict(size=11, color='#333'),
            title='Job Zone (Complexity)',
        ),
        margin=dict(l=80, r=20, t=30, b=60),
        plot_bgcolor='rgba(250,250,250,1)',
    )
    st.plotly_chart(fig, use_container_width=True)

    # Centrality metrics table
    st.subheader("📊 Centrality Metrics")
    st.markdown("Which occupations are most connected (degree), most bridging (betweenness), or most reachable (PageRank)?")

    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G)

    centrality_df = pd.DataFrame({
        'Occupation': [title_lookup.get(c, c) for c in our_codes],
        'Cluster': [cluster_lookup.get(c, '') for c in our_codes],
        'Degree': [degree.get(c, 0) for c in our_codes],
        'Betweenness': [round(betweenness.get(c, 0), 3) for c in our_codes],
        'PageRank': [round(pagerank.get(c, 0), 3) for c in our_codes],
    }).sort_values('Degree', ascending=False)

    st.dataframe(centrality_df, hide_index=True, use_container_width=True)


# ================================================================
# PAGE 3: CAREER MATCH
# ================================================================
elif page == "Career Match":
    st.title("\U0001f3af Career Match Tool")
    st.markdown("Select your skills and experience level to find matching construction careers in Maine.")

    X_raw = data['X_raw']
    skill_cols = [c for c in X_raw.columns if c.startswith('skill_')]
    knowledge_cols = [c for c in X_raw.columns if c.startswith('knowledge_')]

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
        user_skills = {}
        for s in selected_skills:
            user_skills[f'skill_{s}'] = exp_level
        for k in selected_knowledge:
            user_skills[f'knowledge_{k}'] = exp_level

        if len(user_skills) < 2:
            st.warning("Please select at least 2 skills or knowledge areas.")
        else:
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

            # Top 5 results with training info
            top5 = results.head(5)
            codes_with_app = data['apprentice'].dropna(subset=['onet_match'])['onet_match'].unique()

            for i, (code, row) in enumerate(top5.iterrows()):
                col1, col2, col3 = st.columns([3, 2, 1])
                col1.markdown(f"**{i+1}. {row['Occupation']}**")
                has_app = "\u2705" if code in codes_with_app else "\u274c"
                col2.markdown(f"`{row['Cluster']}` | Job Zone {row['Job Zone']} | Apprenticeship: {has_app}")
                col3.metric("Match", f"{row['Match Score']}%")

            st.divider()

            with st.expander("View all occupations ranked"):
                st.dataframe(results, hide_index=True, use_container_width=True)