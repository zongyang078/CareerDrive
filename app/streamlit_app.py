"""
CareerDrive Streamlit Demo — Refactored

Three pages:
1. Industry Dashboard — descriptive stats, company landscape, supply-demand gap
2. Clustering Explorer — interactive PCA scatter plot with cluster details
3. Career Match — input skills, get matched occupations + training pathways

Run with: streamlit run app/streamlit_app.py
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from pyvis.network import Network
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

# --- Job Zone plain-language labels ---
JZ_LABELS = {
    2: 'Entry-level (high school + training)',
    3: 'Mid-level (vocational / associate degree)',
    4: 'Advanced (bachelor\'s / apprenticeship)',
    5: 'Expert (graduate degree / 5+ years)',
}

# --- Career Match: curated skill groups with plain-English labels ---
SKILL_GROUPS = {
    'Working with your hands / on-site': [
        'skill_Equipment Maintenance', 'skill_Equipment Selection', 'skill_Installation',
        'skill_Operation and Control', 'skill_Operations Monitoring', 'skill_Repairing',
    ],
    'Operating vehicles & heavy machinery': [
        'skill_Operation and Control', 'skill_Equipment Maintenance',
        'skill_Operations Monitoring',
    ],
    'Problem-solving & troubleshooting': [
        'skill_Troubleshooting', 'skill_Critical Thinking', 'skill_Complex Problem Solving',
        'skill_Judgment and Decision Making',
    ],
    'Reading plans & technical documents': [
        'knowledge_Building and Construction', 'knowledge_Engineering and Technology',
        'knowledge_Design', 'knowledge_Mathematics',
    ],
    'Leading a team or managing a project': [
        'skill_Management of Personnel Resources', 'skill_Coordination',
        'skill_Instructing', 'skill_Monitoring',
    ],
    'Safety & regulations': [
        'knowledge_Public Safety and Security', 'knowledge_Law and Government',
        'skill_Operations Monitoring',
    ],
    'Computers & technology': [
        'knowledge_Computers and Electronics', 'skill_Technology Design',
        'skill_Programming',
    ],
}

# --- Load Data ---
@st.cache_data
def load_data():
    processed = BASE_DIR / 'data' / 'processed'
    features = pd.read_csv(processed / 'occupation_features.csv', index_col=0)
    clusters = pd.read_csv(processed / 'cluster_labels.csv')
    job_zones = pd.read_csv(processed / 'job_zones.csv')
    related = pd.read_csv(processed / 'related_occupations.csv')
    agc = pd.read_csv(processed / 'agc_members.csv')
    apprentice = pd.read_csv(processed / 'apprenticeships.csv')
    cc = pd.read_csv(processed / 'community_college.csv')
    umaine = pd.read_csv(processed / 'umaine_programs.csv')
    companies = pd.read_csv(processed / 'companies_merged.csv')

    titles = features['Title']
    X_raw = features.drop(columns='Title')
    scaler = StandardScaler().fit(X_raw)
    X_scaled = scaler.transform(X_raw)

    pca = PCA(n_components=2).fit(X_scaled)
    pca2 = pca.transform(X_scaled)

    # Load mappings from config file
    with open(BASE_DIR / 'data' / 'mappings.json') as f:
        mappings = json.load(f)

    apprentice['onet_match'] = apprentice['title'].map(mappings['apprentice_to_onet'])
    cc['cluster_match'] = cc['program_name'].map(mappings['cc_to_cluster']).fillna('Unmatched')
    umaine['cluster_match'] = umaine['program_name'].map(mappings['umaine_to_cluster']).fillna('Unmatched')

    cos_sim = pd.read_csv(processed / 'cosine_similarity.csv', index_col=0)

    return {
        'features': features, 'clusters': clusters, 'job_zones': job_zones,
        'related': related, 'agc': agc, 'apprentice': apprentice,
        'cc': cc, 'umaine': umaine, 'companies': companies,
        'titles': titles, 'X_raw': X_raw, 'scaler': scaler,
        'X_scaled': X_scaled, 'pca': pca, 'pca2': pca2,
        'cos_sim': cos_sim,
    }

try:
    data = load_data()
except FileNotFoundError as e:
    st.error(f"数据文件缺失: {e}\n\n请先按顺序运行 notebooks 01–04 生成处理后的数据。")
    st.stop()
except Exception as e:
    st.error(f"数据加载失败: {e}")
    st.stop()

# --- Sidebar Navigation ---
st.sidebar.title("\U0001f6e4\ufe0f CareerDrive")
st.sidebar.markdown("*Maine Construction Industry Career Map*")
page = st.sidebar.radio("Navigate", ["Industry Dashboard", "Career Path Map", "Career Match"])
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
        st.caption("Zone 2: High school + short training · Zone 3: Vocational/Associate · Zone 4: Bachelor's/Apprenticeship · Zone 5: Graduate/Expert")
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
# PAGE 2: CAREER PATH MAP
# ================================================================
elif page == "Career Path Map":
    st.title("🗺️ Career Path Map")
    st.markdown(
        "Explore how construction careers connect to each other. "
        "**Drag nodes** to rearrange, **hover** over any occupation to see details, "
        "and use the scroll wheel to zoom."
    )
    st.markdown(
        "Each node is an occupation — size reflects experience required. "
        "Lines show which roles are closely related by skills or official O\\*NET pathways."
    )

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
    cos_sim = data['cos_sim']
    for i, c1 in enumerate(our_codes):
        for j, c2 in enumerate(our_codes):
            if i >= j:
                continue
            if c1 in cos_sim.index and c2 in cos_sim.columns:
                if cos_sim.loc[c1, c2] >= 0.95 and not G.has_edge(c1, c2):
                    G.add_edge(c1, c2, source='similarity')

    # Job Zone descriptions for tooltips
    jz_desc = {
        2: 'Zone 2 — Some preparation (high school + short-term training)',
        3: 'Zone 3 — Medium preparation (vocational / associate degree)',
        4: 'Zone 4 — Considerable preparation (bachelor\'s degree / apprenticeship)',
        5: 'Zone 5 — Extensive preparation (graduate degree / years of experience)',
    }

    # Build pyvis network
    net = Network(height='650px', width='100%', bgcolor='#f9f9f9', font_color='#333')
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120, spring_strength=0.05)

    for code in our_codes:
        cname = cluster_lookup.get(code, 'Skilled Trades')
        jz = zone_lookup.get(code, 3)
        title_str = title_lookup.get(code, code)
        short_title = title_str.split(',')[0]
        tooltip = (
            f"{title_str}\n"
            f"Cluster: {cname}\n"
            f"{jz_desc.get(jz, f'Zone {jz}')}\n"
            f"Connections: {G.degree(code)}"
        )
        net.add_node(
            code,
            label=short_title,
            title=tooltip,
            color=CLUSTER_COLORS[cname],
            size=jz * 6 + 8,
            font={'size': 11, 'color': '#222'},
            borderWidth=2,
            borderWidthSelected=4,
        )

    for src, dst, edata in G.edges(data=True):
        is_onet = edata.get('source') == 'onet'
        net.add_edge(
            src, dst,
            color='rgba(80,80,80,0.5)' if is_onet else 'rgba(180,180,180,0.3)',
            width=2.0 if is_onet else 0.8,
            title='O*NET Related Occupation' if is_onet else 'High Skill Similarity (≥0.95)',
            dashes=not is_onet,
        )

    # Legend via HTML injection
    legend_html = ''.join([
        f'<div style="display:inline-block;margin-right:18px">'
        f'<span style="display:inline-block;width:14px;height:14px;border-radius:50%;'
        f'background:{color};margin-right:5px;vertical-align:middle"></span>'
        f'<span style="font-size:13px">{name}</span></div>'
        for name, color in CLUSTER_COLORS.items()
    ])
    legend_html += (
        '<div style="margin-top:6px;font-size:12px;color:#666">'
        '<b>─── </b>O*NET Related Occupation &nbsp;&nbsp;'
        '<b style="letter-spacing:2px">- - -</b> High Skill Similarity (≥0.95) &nbsp;&nbsp;'
        'Node size = Job Zone complexity</div>'
    )
    st.markdown(legend_html, unsafe_allow_html=True)

    net_html = net.generate_html()
    components.html(net_html, height=660, scrolling=False)


# ================================================================
# PAGE 3: CAREER MATCH
# ================================================================
elif page == "Career Match":
    st.title("🎯 Find Your Career in Maine Construction")
    st.markdown(
        "Answer a few questions about your background and strengths. "
        "We'll match you to construction careers and show available training pathways."
    )

    X_raw = data['X_raw']

    st.subheader("1. What best describes your work background?")
    selected_groups = st.multiselect(
        "Select all that apply:",
        options=list(SKILL_GROUPS.keys()),
        default=['Working with your hands / on-site'],
        help="Pick the areas where you have experience or feel most confident."
    )

    st.subheader("2. How would you rate your overall experience level?")
    exp_labels = {
        1.0: "Just starting out — no prior experience",
        2.0: "Some experience — entry-level or training",
        3.0: "Solid experience — a few years on the job",
        4.0: "Experienced — strong background in the field",
        5.0: "Expert — many years, may supervise others",
    }
    exp_level = st.select_slider(
        "Experience level:",
        options=[1.0, 2.0, 3.0, 4.0, 5.0],
        value=2.0,
        format_func=lambda x: exp_labels[x],
    )

    st.subheader("3. Are you open to further training or education?")
    open_to_training = st.radio(
        "",
        options=["Yes, I'm open to apprenticeships or programs", "I prefer roles I can enter right away"],
        index=0,
        horizontal=True,
    )

    if st.button("Find Matching Careers", type="primary"):
        # Build feature vector from selected groups
        user_skills = {}
        for group in selected_groups:
            for feat in SKILL_GROUPS.get(group, []):
                if feat in X_raw.columns:
                    user_skills[feat] = exp_level

        if len(user_skills) < 2:
            st.warning("Please select at least one background area above.")
        else:
            valid_features = list(user_skills.keys())
            X_sub = X_raw[valid_features]
            user_vec = np.array([user_skills[f] for f in valid_features]).reshape(1, -1)

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

            # Filter by training preference
            codes_with_app = data['apprentice'].dropna(subset=['onet_match'])['onet_match'].unique()
            if open_to_training == "I prefer roles I can enter right away":
                results = results[results['Job Zone'] <= 3]

            st.divider()
            st.subheader("Your Top Career Matches")
            st.caption("Based on your background and experience level.")

            top5 = results.head(5)
            for i, (code, row) in enumerate(top5.iterrows()):
                has_app = code in codes_with_app
                jz_label = JZ_LABELS.get(int(row['Job Zone']), f"Zone {row['Job Zone']}")

                with st.container(border=True):
                    c1, c2, c3 = st.columns([3, 3, 1])
                    c1.markdown(f"**{i+1}. {row['Occupation']}**")
                    c1.caption(jz_label)
                    c2.markdown(f"**Career track:** {row['Cluster']}")
                    c2.markdown(
                        f"**Apprenticeship available:** {'✅ Yes' if has_app else '❌ Not in current data'}"
                    )
                    c3.metric("Match", f"{row['Match Score']}%")

            st.divider()
            with st.expander("See all 20 occupations ranked"):
                display_results = results.copy()
                display_results['Experience Required'] = display_results['Job Zone'].map(JZ_LABELS)
                display_results['Apprenticeship'] = display_results.index.map(
                    lambda c: '✅' if c in codes_with_app else '❌'
                )
                st.dataframe(
                    display_results[['Occupation', 'Career Track'.replace('Career Track', 'Cluster'),
                                     'Experience Required', 'Apprenticeship', 'Match Score']].rename(
                        columns={'Cluster': 'Career Track'}
                    ),
                    hide_index=True, use_container_width=True
                )