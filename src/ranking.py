import pandas as pd
import numpy as np



# Normalization


def min_max_scale(series):

    series = series.fillna(0)

    if series.max() == series.min():
        return pd.Series(0.0, index=series.index)

    return (series - series.min()) / (
        series.max() - series.min()
    )



# Technical Score


def build_technical_score(features):

    df = features.copy()

    technical = pd.DataFrame()

    technical["candidate_id"] = df["candidate_id"]

    technical["technical_score"] = (

        0.25 * min_max_scale(df["ai_skill_score"]) +

        0.25 * min_max_scale(df["production_score"]) +

        0.20 * min_max_scale(df["career_quality_score"]) +

        0.15 * min_max_scale(df["python_endorsements"]) +

        0.15 * min_max_scale(df["embedding_skill_count"])

    )

    return technical



# Product Fit Score


def build_product_fit_score(features):

    df = features.copy()

    score = np.zeros(len(df))

    ai_titles = [

        "AI Engineer",
        "Applied Scientist",
        "Machine Learning",
        "ML Engineer",
        "Recommendation",
        "Search",
        "Retrieval",
        "Ranking",
        "NLP"

    ]

    title_match = df["current_title"].str.contains(

        "|".join(ai_titles),

        case=False,

        na=False

    )

    score += title_match.astype(int) * 0.35

    score += min_max_scale(df["product_company_ratio"]) * 0.35

    score += min_max_scale(df["retrieval_mentions"]) * 0.15

    score += min_max_scale(df["vector_db_skill_count"]) * 0.15

    # Penalize candidates with mostly service-company experience
    score -= min_max_scale(df["service_company_ratio"]) * 0.10

    score = np.clip(score, 0, 1)

    return pd.DataFrame({

        "candidate_id": df["candidate_id"],

        "product_fit_score": score

    })



# Experience Fit Score


def build_experience_fit_score(features):

    df = features.copy()

    experience = df["years_of_experience"]

    score = np.where(
        (experience >= 6) & (experience <= 8),
        1.0,
        np.where(
            (experience >= 5) & (experience <= 9),
            0.9,
            np.where(
                (experience >= 4) & (experience <= 10),
                0.7,
                0.3
            )
        )
    )

    return pd.DataFrame({
        "candidate_id": df["candidate_id"],
        "experience_fit_score": score
    })



# Final Ranking Score


def build_final_score(

    ranking_df,
    technical_score,
    product_fit,
    experience_fit

):

    df = (

        ranking_df

        .merge(
            technical_score,
            on="candidate_id"
        )

        .merge(
            product_fit,
            on="candidate_id"
        )

        .merge(
            experience_fit,
            on="candidate_id"
        )

    )

    # Normalize

    df["semantic_similarity"] = min_max_scale(
        df["semantic_similarity"]
    )

    df["technical_score"] = min_max_scale(
        df["technical_score"]
    )

    df["behavior_score"] = min_max_scale(
        df["behavior_score"]
    )

    df["product_fit_score"] = min_max_scale(
        df["product_fit_score"]
    )

    df["experience_fit_score"] = min_max_scale(
        df["experience_fit_score"]
    )

    # Final weighted score

    df["final_score"] = (

        0.35 * df["semantic_similarity"] +

        0.30 * df["technical_score"] +

        0.15 * df["behavior_score"] +

        0.10 * df["product_fit_score"] +

        0.10 * df["experience_fit_score"]

    )

    # Cleanup duplicate columns

    df = df.drop(columns=["trust_score_x"], errors="ignore")

    df = df.rename(

        columns={

            "trust_score_y": "trust_score"

        }

    )

    return df



# Ranking


def rank_candidates(df):

    return (

        df

        .sort_values(

            "final_score",

            ascending=False

        )

        .reset_index(drop=True)

    )



# Submission Helper


def build_submission_features(df):

    keep = [

        "candidate_id",

        "current_title",

        "years_of_experience",

        "semantic_similarity",

        "technical_score",

        "behavior_score",

        "product_fit_score",

        "experience_fit_score",

        "final_score"

    ]

    return df[keep].copy()