"""
Data cleaning and merging utilities for CareerDrive project.
"""

import pandas as pd


def load_agc_members(excel_path: str) -> pd.DataFrame:
    """Load and clean AGC Member List from Excel."""
    df = pd.read_excel(excel_path, sheet_name="AGC Member list ", header=3)
    df.columns = ["type", "member_name"]
    df = df.dropna(subset=["member_name"])
    df["type"] = df["type"].ffill()  # forward fill category labels
    df = df[df["type"].isin([
        "General Contractors", "Specialty Contractors",
        "Suppliers", "Service Providers", "Developer"
    ])]
    return df.reset_index(drop=True)


def load_dot_prequal(excel_path: str) -> pd.DataFrame:
    """Load and clean DOT Prequal contractor list."""
    df = pd.read_excel(excel_path, sheet_name="JobBoards -DOT Prequal List")
    df.columns = ["company_name", "website"]
    df = df.dropna(subset=["company_name"])
    return df.reset_index(drop=True)


def load_apprenticeships(excel_path: str) -> pd.DataFrame:
    """Load AGC Sponsored Apprenticeships data."""
    df = pd.read_excel(excel_path, sheet_name="AGC Sponsored Apprenticeships ", header=1)
    df.columns = ["onet_code", "title", "term_hours"]
    df = df.dropna(subset=["title"])
    return df.reset_index(drop=True)


def load_community_college(excel_path: str) -> pd.DataFrame:
    """Load Community College Programs data."""
    df = pd.read_excel(excel_path, sheet_name="Community College Programs", header=1)
    df.columns = ["college", "program_name", "credentials", "link"]
    df = df.dropna(subset=["program_name"])
    return df.reset_index(drop=True)


def load_umaine(excel_path: str) -> pd.DataFrame:
    """Load UMaine Programs data."""
    df = pd.read_excel(excel_path, sheet_name="Umaine Programs", header=2)
    df.columns = ["campus", "program_category", "degree_type", "program_name", "link"]
    df = df.dropna(subset=["program_name"])
    return df.reset_index(drop=True)


def load_onet_codes(excel_path: str) -> pd.DataFrame:
    """Load Construction O*NET Codes with titles and descriptions."""
    df = pd.read_excel(excel_path, sheet_name="Construction ONet Codes", header=2)
    df.columns = ["onet_code", "title", "key_role"]
    df = df.dropna(subset=["onet_code"])
    return df.reset_index(drop=True)


def merge_company_lists(agc_df: pd.DataFrame, dot_df: pd.DataFrame) -> pd.DataFrame:
    """Merge AGC members and DOT prequal lists, removing duplicates."""
    # Normalize names for matching
    agc_df["name_clean"] = agc_df["member_name"].str.strip().str.lower()
    dot_df["name_clean"] = dot_df["company_name"].str.strip().str.lower()

    # Find DOT companies not already in AGC list
    new_companies = dot_df[~dot_df["name_clean"].isin(agc_df["name_clean"])]

    # Combine
    combined = pd.concat([
        agc_df[["member_name", "type"]].rename(columns={"member_name": "company_name"}),
        new_companies[["company_name"]].assign(type="DOT Prequal")
    ], ignore_index=True)

    return combined
