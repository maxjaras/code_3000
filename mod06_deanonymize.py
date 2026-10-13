import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    quasi_identifiers = [
        col for col in anon_df.columns if col in aux_df.columns and col != "anon_id"
    ]

    merged = pd.merge(anon_df, aux_df, on=quasi_identifiers, how="inner")

    match_counts = merged["anon_id"].value_counts()
    unique_anon_ids = match_counts[match_counts == 1].index

    unique_matches = merged[merged["anon_id"].isin(unique_anon_ids)]

    name_col = "name" if "name" in aux_df.columns else "matched_name"

    result = unique_matches[["anon_id", name_col]].rename(
        columns={name_col: "matched_name"}
    )

    return result


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """

    if len(anon_df) == 0:
        return 0.0
    
    return len(matches_df) / len(anon_df)   

