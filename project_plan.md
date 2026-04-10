# CareerDrive Project Plan

DS 5230 Final Project — Spring 2026

## 1. Project Overview

CareerDrive is a collaboration between MaineDOT, AGC Maine, and Northeastern's Roux Institute.
The client wants an AI-driven career portal for Maine's construction industry. Our role is to
analyze their data using unsupervised learning and deliver a career map prototype.

**Deliverables**: Presentation (Apr 16) + Report (Apr 23) + GitHub repo with reproducible code.

## 2. Data Sources

| Source | Records | Description |
|--------|---------|-------------|
| O\*NET 30.2 Database | 20 occupations × 120 features | Skills, Knowledge, Abilities (local text files) |
| O\*NET API v2 | 20 occupations | Tasks, work activities, technology (text enrichment) |
| Client Excel | 8 sheets | AGC members, DOT contractors, apprenticeships, education programs |
| Merged Companies | 272 unique firms | AGC (222) + DOT (57), deduplicated |

From 28 Excel rows → 21 unique O\*NET codes → 20 usable occupations (13-1082.00 has no data).

## 3. Analysis

### Layer 1: EDA (Notebook 03)
- Company distribution by type, occupation overview, education/apprenticeship mapping
- Supply-demand gap analysis: training coverage per cluster

### Layer 2: Unsupervised Learning

**Track A — Skills Clustering (Notebook 04)**
- StandardScaler → PCA (PC1=53.6%) → K-Means k=3 (silhouette=0.315) + DBSCAN comparison
- 3 clusters: Management/Engineering (4), Skilled Trades (6), Entry Level/Operators (10)
- Validation: clusters align with Job Zones (4-5, 2-3, 2 respectively)

**Track B — Text Clustering (Notebook 06)**
- TF-IDF (200 features, bigrams) on API-enriched texts (avg 1417 chars/occupation)
- K-Means silhouette=0.103; NMF 3 topics: Engineering & Management, Installation & Fabrication, Equipment Operations
- Management/Engineering 4/4 perfect match with Track A; 65% overall agreement (ARI=0.269)

**Network Analysis (Notebook 05)**
- Graph: 20 nodes, 51 O\*NET edges, density=0.268, single connected component
- Centrality: hub = Structural Iron Workers (degree=10), bridge = Electricians (betweenness=0.244)
- Louvain: 3 communities, ARI=0.279 vs K-Means; Management/Engineering 100% agreement

### Layer 3: Recommendation Prototype (Notebook 04 + Streamlit Page 3)
- Cosine similarity matching on user-selected skills → top-N occupations + training pathways

## 4. Streamlit Demo

| Page | Content |
|------|---------|
| Industry Dashboard | Company landscape, cluster/job zone distributions, supply-demand gap |
| Clustering Explorer | PCA scatter, cluster details, radar charts, career path network graph |
| Career Match | Skill input → matched occupations with apprenticeship/education info |

## 5. Division of Work

| Member | Data | Analysis | Notebooks | Presentation |
|--------|------|----------|-----------|-------------|
| Person A | O\*NET features + API text | Track A clustering, Track B text clustering, recommendation prototype | 01, 04, 06 | Clustering + text analysis results |
| Person B | Company merge | Network graph, centrality, Louvain | 02, 05 | Network analysis findings |
| Person C | Education/apprenticeship mapping | EDA, supply-demand gap | 03 | Data overview + gap analysis |
| Joint | — | Streamlit integration, slides, report | app/ | Intro, future work, client suggestions |

## 6. Timeline

| Date | Milestone |
|------|-----------|
| ~~Mar~~ | Data collection, feature matrix, clustering (Track A) |
| ~~Early Apr~~ | EDA, network analysis, text clustering (Track B), Streamlit |
| **Apr 16** | **Presentation (10-15 min, in-person)** |
| **Apr 23** | **Report + final GitHub repo** |
