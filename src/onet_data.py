"""
O*NET Database loader — builds feature matrices from the local O*NET 30.2
tab-delimited text files (downloaded from onetcenter.org).

Usage:
    from src.onet_data import build_feature_matrix, load_related_occupations

    codes = ["11-9021.00", "47-2031.00"]
    features = build_feature_matrix("data/raw/onet_db", codes)
    edges = load_related_occupations("data/raw/onet_db", codes)
"""

import re
import pandas as pd


def get_valid_codes(excel_path: str) -> list[str]:
    """
    Extract valid O*NET codes from the project Excel file.

    Filters out section headers, blank rows, and repeated column headers.
    Returns deduplicated list of codes matching the XX-XXXX.XX format.
    """
    df = pd.read_excel(
        excel_path,
        sheet_name="Construction ONet Codes",
        header=2,
    )
    df.columns = ["onet_code", "title", "key_role"]

    # Keep only rows where onet_code matches the XX-XXXX.XX pattern
    pattern = re.compile(r"^\d{2}-\d{4}\.\d{2}$")
    valid = df[df["onet_code"].astype(str).str.match(pattern, na=False)]

    # Deduplicate (11-9021.00 appears in both Building and Utility sections)
    codes = valid["onet_code"].drop_duplicates().tolist()
    return [str(c).strip() for c in codes]


def build_feature_matrix(onet_db_dir: str, codes: list[str]) -> pd.DataFrame:
    """
    Build a feature matrix from the local O*NET database text files.

    Args:
        onet_db_dir: Path to the extracted O*NET database folder
                     (contains Skills.txt, Knowledge.txt, Abilities.txt, etc.)
        codes: List of O*NET-SOC codes to include.

    Returns:
        DataFrame with O*NET-SOC Code as index, Title as first column,
        then skill_*, knowledge_*, ability_* feature columns.
        Uses the Importance (IM) scale values.
    """
    frames = []
    for filename, prefix in [
        ("Skills.txt", "skill"),
        ("Knowledge.txt", "knowledge"),
        ("Abilities.txt", "ability"),
    ]:
        filepath = f"{onet_db_dir}/{filename}"
        df = pd.read_csv(filepath, sep="\t")

        # Filter to Importance scale and our codes
        filtered = df[
            (df["Scale ID"] == "IM") & (df["O*NET-SOC Code"].isin(codes))
        ]

        # Pivot: one row per occupation, one column per element
        pivot = filtered.pivot_table(
            index="O*NET-SOC Code",
            columns="Element Name",
            values="Data Value",
            aggfunc="first",
        )
        pivot.columns = [f"{prefix}_{col}" for col in pivot.columns]
        frames.append(pivot)

    # Combine all features
    feature_matrix = pd.concat(frames, axis=1)

    # Add occupation titles
    occ_data = pd.read_csv(f"{onet_db_dir}/Occupation Data.txt", sep="\t")
    titles = occ_data[occ_data["O*NET-SOC Code"].isin(codes)][
        ["O*NET-SOC Code", "Title"]
    ].set_index("O*NET-SOC Code")

    feature_matrix = feature_matrix.join(titles)

    # Reorder: Title first, then features
    cols = ["Title"] + [c for c in feature_matrix.columns if c != "Title"]
    feature_matrix = feature_matrix[cols]

    return feature_matrix


def load_related_occupations(
    onet_db_dir: str, codes: list[str]
) -> pd.DataFrame:
    """
    Load O*NET Related Occupations edges where both endpoints are in our code list.

    Returns DataFrame with columns:
        O*NET-SOC Code, Related O*NET-SOC Code, Relatedness Tier, Index
    """
    df = pd.read_csv(f"{onet_db_dir}/Related Occupations.txt", sep="\t")
    return df[
        df["O*NET-SOC Code"].isin(codes)
        & df["Related O*NET-SOC Code"].isin(codes)
    ].reset_index(drop=True)


def load_job_zones(onet_db_dir: str, codes: list[str]) -> pd.DataFrame:
    """
    Load Job Zone (complexity level 1-5) for each occupation.
    """
    df = pd.read_csv(f"{onet_db_dir}/Job Zones.txt", sep="\t")
    return df[df["O*NET-SOC Code"].isin(codes)][
        ["O*NET-SOC Code", "Job Zone"]
    ].reset_index(drop=True)


if __name__ == "__main__":
    import sys

    excel = "data/raw/Career_Drive_Project_Data_Sources.xlsx"
    db = "data/raw/onet_db"

    codes = get_valid_codes(excel)
    print(f"Valid codes from Excel: {len(codes)}")

    features = build_feature_matrix(db, codes)
    print(f"Feature matrix: {features.shape}")
    print(f"Missing values: {features.drop(columns='Title').isna().sum().sum()}")

    edges = load_related_occupations(db, codes)
    print(f"Related occupation edges: {len(edges)}")

    jz = load_job_zones(db, codes)
    print(f"Job zones: {len(jz)}")
