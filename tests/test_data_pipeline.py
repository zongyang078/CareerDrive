"""
Smoke tests for the CareerDrive data pipeline.
Validates that all processed CSVs exist, have expected shape, and contain no
critical missing values. Run with: pytest tests/
"""

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def features():
    return pd.read_csv(PROCESSED / "occupation_features.csv", index_col=0)


@pytest.fixture(scope="module")
def clusters():
    return pd.read_csv(PROCESSED / "cluster_labels.csv")


@pytest.fixture(scope="module")
def mappings():
    with open(ROOT / "data" / "mappings.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CSV existence & shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename,min_rows", [
    ("occupation_features.csv", 20),
    ("cluster_labels.csv", 20),
    ("job_zones.csv", 20),
    ("related_occupations.csv", 1),
    ("agc_members.csv", 200),
    ("apprenticeships.csv", 19),
    ("community_college.csv", 14),
    ("umaine_programs.csv", 10),
    ("companies_merged.csv", 250),
    ("cosine_similarity.csv", 20),
])
def test_csv_exists_and_has_rows(filename, min_rows):
    path = PROCESSED / filename
    assert path.exists(), f"Missing file: {filename}"
    df = pd.read_csv(path, index_col=0 if filename in ("occupation_features.csv", "cosine_similarity.csv") else None)
    assert len(df) >= min_rows, f"{filename}: expected >={min_rows} rows, got {len(df)}"


# ---------------------------------------------------------------------------
# occupation_features integrity
# ---------------------------------------------------------------------------

def test_features_no_missing(features):
    assert features.isnull().sum().sum() == 0, "occupation_features.csv has missing values"


def test_features_has_title_column(features):
    assert "Title" in features.columns


def test_features_exactly_20_occupations(features):
    assert len(features) == 20


# ---------------------------------------------------------------------------
# cluster_labels integrity
# ---------------------------------------------------------------------------

def test_clusters_required_columns(clusters):
    for col in ["O*NET-SOC Code", "Title", "cluster_name", "Job Zone"]:
        assert col in clusters.columns, f"Missing column: {col}"


def test_clusters_valid_names(clusters):
    valid = {"Management/Engineering", "Skilled Trades", "Entry Level/Operators"}
    assert set(clusters["cluster_name"].unique()).issubset(valid)


def test_clusters_codes_match_features(features, clusters):
    feature_codes = set(features.index)
    cluster_codes = set(clusters["O*NET-SOC Code"])
    assert feature_codes == cluster_codes, "O*NET codes differ between features and clusters"


# ---------------------------------------------------------------------------
# cosine_similarity integrity
# ---------------------------------------------------------------------------

def test_cosine_similarity_shape():
    cos = pd.read_csv(PROCESSED / "cosine_similarity.csv", index_col=0)
    assert cos.shape == (20, 20), f"Expected 20×20, got {cos.shape}"


def test_cosine_similarity_diagonal_ones():
    cos = pd.read_csv(PROCESSED / "cosine_similarity.csv", index_col=0)
    diag = [cos.iloc[i, i] for i in range(len(cos))]
    assert all(abs(v - 1.0) < 1e-6 for v in diag), "Diagonal of cosine similarity matrix should be 1.0"


# ---------------------------------------------------------------------------
# community_college integrity (post-cleanup)
# ---------------------------------------------------------------------------

def test_community_college_no_mixed_rows():
    cc = pd.read_csv(PROCESSED / "community_college.csv")
    assert set(cc.columns) == {"college", "program_name", "credentials", "link"}, \
        "community_college.csv has unexpected columns (mixed table rows?)"
    assert cc["college"].isnull().sum() == 0, "college column has missing values"


# ---------------------------------------------------------------------------
# mappings.json integrity
# ---------------------------------------------------------------------------

def test_mappings_has_required_keys(mappings):
    for key in ("apprentice_to_onet", "cc_to_cluster", "umaine_to_cluster"):
        assert key in mappings, f"mappings.json missing key: {key}"


def test_cc_mapping_covers_all_programs(mappings):
    cc = pd.read_csv(PROCESSED / "community_college.csv")
    mapping = mappings["cc_to_cluster"]
    unmatched = [p for p in cc["program_name"] if p not in mapping]
    assert unmatched == [], f"Programs not in cc_to_cluster mapping: {unmatched}"


def test_umaine_mapping_covers_all_programs(mappings):
    umaine = pd.read_csv(PROCESSED / "umaine_programs.csv")
    mapping = mappings["umaine_to_cluster"]
    unmatched = [p for p in umaine["program_name"] if p not in mapping]
    assert unmatched == [], f"Programs not in umaine_to_cluster mapping: {unmatched}"


def test_apprentice_mapping_covers_all_titles(mappings):
    apprentice = pd.read_csv(PROCESSED / "apprenticeships.csv")
    mapping = mappings["apprentice_to_onet"]
    unmatched = [t for t in apprentice["title"] if t not in mapping]
    assert unmatched == [], f"Apprenticeships not in mapping: {unmatched}"
