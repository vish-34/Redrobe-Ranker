import pandas as pd
import numpy as np
import re

# profile feature

def build_experience_features(profiles_df):
    experience_df = profiles_df[
        [
            "candidate_id",
            "years_of_experience",
        ]
    ].copy()

    return experience_df


def build_title_features(profiles_df):
    title_df = profiles_df[
        [
            "candidate_id",
            "current_title",
        ]
    ].copy()

    return title_df

# career feature

def build_career_stability_features(career_df):
    stability = (
        career_df
        .groupby("candidate_id")
        .agg(
            total_jobs = ("company", 'count'),
            unique_companies = ("company", "nunique"),
            avg_job_duration = ("duration_months","mean"),
            shortest_job = ("duration_months", "min"),
            longest_job = ("duration_months", "max")
            )
            .reset_index()
    )
    return stability

def build_product_company_features(career_df):
    """
    Computes product-company and service-company experience ratios.
    """

    PRODUCT_COMPANIES = {
        "Meta",
        "Google",
        "Flipkart",
        "Swiggy",
        "Razorpay",
        "CRED",
        "Meesho",
        "Zoho",
        "Freshworks",
        "Nykaa",
        "Paytm",
    }

    SERVICE_COMPANIES = {
        "Infosys",
        "TCS",
        "Wipro",
        "Accenture",
        "Capgemini",
        "Cognizant",
        "HCL",
        "Tech Mahindra",
        "Mindtree",
        "Mphasis",
    }

    df = career_df.copy()

    df["is_product_company"] = df["company"].isin(PRODUCT_COMPANIES).astype(int)
    df["is_service_company"] = df["company"].isin(SERVICE_COMPANIES).astype(int)

    company_features = (
        df.groupby("candidate_id")
        .agg(
            product_company_ratio=("is_product_company", "mean"),
            service_company_ratio=("is_service_company", "mean"),
            product_company_count=("is_product_company", "sum"),
            service_company_count=("is_service_company", "sum"),
        )
        .reset_index()
    )

    return company_features

def build_career_description_features(career_df):

    KEYWORDS = {
        "retrieval": [
            "retrieval",
            "information retrieval",
            "semantic search"
        ],

        "ranking": [
            "ranking",
            "learning to rank",
            "ltr"
        ],

        "recommendation": [
            "recommendation",
            "recommendation system",
            "recommender"
        ],

        "embeddings": [
            "embedding",
            "embeddings",
            "sentence transformer"
        ],

        "vector_db": [
            "faiss",
            "pinecone",
            "milvus",
            "qdrant",
            "weaviate"
        ],

        "evaluation": [
            "ndcg",
            "mrr",
            "precision",
            "recall",
            "offline evaluation",
            "a/b testing"
        ],

        "production": [
            "production",
            "deployment",
            "deployed",
            "monitoring"
        ]
    }

    df = career_df.copy()

    df["description"] = (
        df["description"]
        .fillna("")
        .str.lower()
    )

    rows = []

    for candidate_id, group in df.groupby("candidate_id"):

        text = " ".join(group["description"])

        row = {
            "candidate_id": candidate_id
        }

        for feature, words in KEYWORDS.items():

            count = 0

            for word in words:
                count += len(re.findall(re.escape(word), text))

            row[f"{feature}_mentions"] = count

        rows.append(row)

    return pd.DataFrame(rows)

def build_career_growth_features(career_df):
    pass

def build_retrieval_features(career_df):
    pass


# skill feature

def build_python_features(skill_df):

    PROFICIENCY_MAP = {
        "beginner": 1,
        "intermediate": 2,
        "advanced": 3,
        "expert": 4
    }

    df = skill_df.copy()

    df = df[df["skill_name"].str.lower() == "python"]

    df["python_proficiency"] = (
        df["proficiency"]
        .str.lower()
        .map(PROFICIENCY_MAP)
    )

    python_df = (
        df.groupby("candidate_id")
        .agg(
            python_duration=("duration_months", "max"),
            python_endorsements=("endorsements", "max"),
            python_proficiency=("python_proficiency", "max")
        )
        .reset_index()
    )

    return python_df


def build_embedding_features(skill_df):

    EMBEDDING_SKILLS = {
        "Embeddings",
        "Sentence Transformers",
        "LoRA",
        "QLoRA",
        "PEFT"
    }

    df = skill_df.copy()

    df = df[df["skill_name"].isin(EMBEDDING_SKILLS)]

    embedding_df = (
        df.groupby("candidate_id")
        .agg(
            embedding_skill_count=("skill_name", "count"),
            embedding_duration=("duration_months", "sum"),
            embedding_endorsements=("endorsements", "sum")
        )
        .reset_index()
    )

    return embedding_df


def build_vector_database_features(skill_df):

    VECTOR_DB = {
        "FAISS",
        "Pinecone",
        "Milvus",
        "Qdrant",
        "Weaviate"
    }

    df = skill_df.copy()

    df = df[df["skill_name"].isin(VECTOR_DB)]

    vector_df = (
        df.groupby("candidate_id")
        .agg(
            vector_db_skill_count=("skill_name", "count"),
            vector_db_duration=("duration_months", "sum"),
            vector_db_endorsements=("endorsements", "sum")
        )
        .reset_index()
    )

    return vector_df


def build_skill_summary(skill_df):

    summary = (
        skill_df
        .groupby("candidate_id")
        .agg(
            total_skills=("skill_name", "count"),
            avg_skill_duration=("duration_months", "mean"),
            total_endorsements=("endorsements", "sum")
        )
        .reset_index()
    )

    return summary


# bahaviour feature

def build_behavior_features(signals_df):

    df = signals_df.copy()

    # Convert booleans to integers
    bool_cols = [
        "open_to_work_flag",
        "willing_to_relocate",
        "verified_email",
        "verified_phone",
        "linkedin_connected"
    ]

    for col in bool_cols:
        df[col] = df[col].astype(int)

    # Encode preferred work mode
    work_mode_map = {
        "onsite": 0,
        "hybrid": 1,
        "flexible": 2,
        "remote": 3
    }

    df["preferred_work_mode"] = (
        df["preferred_work_mode"]
        .str.lower()
        .map(work_mode_map)
        .fillna(-1)
        .astype(int)
    )

    # Extract salary range
    df["expected_salary_min"] = df["expected_salary_range_inr_lpa"].apply(
        lambda x: x.get("min", np.nan) if isinstance(x, dict) else np.nan
    )

    df["expected_salary_max"] = df["expected_salary_range_inr_lpa"].apply(
        lambda x: x.get("max", np.nan) if isinstance(x, dict) else np.nan
    )

    # Count completed skill assessments
    df["num_skill_assessments"] = df["skill_assessment_scores"].apply(
        lambda x: len(x) if isinstance(x, dict) else 0
    )

    behavior_df = df[
        [
            "candidate_id",

            "profile_completeness_score",

            "open_to_work_flag",

            "profile_views_received_30d",

            "applications_submitted_30d",

            "recruiter_response_rate",

            "avg_response_time_hours",

            "connection_count",

            "endorsements_received",

            "notice_period_days",

            "preferred_work_mode",

            "willing_to_relocate",

            "github_activity_score",

            "search_appearance_30d",

            "saved_by_recruiters_30d",

            "interview_completion_rate",

            "offer_acceptance_rate",

            "verified_email",

            "verified_phone",

            "linkedin_connected",

            "expected_salary_min",

            "expected_salary_max",

            "num_skill_assessments"
        ]
    ].copy()

    return behavior_df

    df = signals_df.copy()

    # Convert booleans to integers
    bool_cols = [
        "open_to_work_flag",
        "willing_to_relocate",
        "verified_email",
        "verified_phone",
        "linkedin_connected"
    ]

    for col in bool_cols:
        df[col] = df[col].astype(int)

    # Extract salary range
    df["expected_salary_min"] = df["expected_salary_range_inr_lpa"].apply(
        lambda x: x.get("min", np.nan) if isinstance(x, dict) else np.nan
    )

    df["expected_salary_max"] = df["expected_salary_range_inr_lpa"].apply(
        lambda x: x.get("max", np.nan) if isinstance(x, dict) else np.nan
    )

    # Count completed skill assessments
    df["num_skill_assessments"] = df["skill_assessment_scores"].apply(
        lambda x: len(x) if isinstance(x, dict) else 0
    )

    behavior_df = df[
        [
            "candidate_id",

            "profile_completeness_score",

            "open_to_work_flag",

            "profile_views_received_30d",

            "applications_submitted_30d",

            "recruiter_response_rate",

            "avg_response_time_hours",

            "connection_count",

            "endorsements_received",

            "notice_period_days",

            "preferred_work_mode",

            "willing_to_relocate",

            "github_activity_score",

            "search_appearance_30d",

            "saved_by_recruiters_30d",

            "interview_completion_rate",

            "offer_acceptance_rate",

            "verified_email",

            "verified_phone",

            "linkedin_connected",

            "expected_salary_min",

            "expected_salary_max",

            "num_skill_assessments"
        ]
    ].copy()

    return behavior_df

#extra features
def build_ai_skill_score(features):

    df = features.copy()

    df["ai_skill_score"] = (
        3 * df["python_proficiency"] +
        2 * df["python_endorsements"] / 10 +
        4 * df["embedding_skill_count"] +
        3 * df["vector_db_skill_count"] +
        2 * df["retrieval_mentions"] +
        2 * df["ranking_mentions"] +
        2 * df["recommendation_mentions"] +
        2 * df["embeddings_mentions"] +
        2 * df["vector_db_mentions"]
    )

    return df[["candidate_id", "ai_skill_score"]]

def build_production_score(features):

    df = features.copy()

    df["production_score"] = (
        2 * df["years_of_experience"] +
        2 * df["avg_job_duration"] / 12 +
        4 * df["production_mentions"] +
        3 * df["product_company_count"]
    )

    return df[["candidate_id", "production_score"]]

def build_career_quality_score(features):

    df = features.copy()

    df["career_quality_score"] = (
        2 * df["avg_job_duration"] / 12 +
        2 * df["product_company_ratio"] +
        df["unique_companies"] -
        0.5 * df["total_jobs"]
    )

    return df[["candidate_id", "career_quality_score"]]

def build_recruiter_score(features):

    df = features.copy()

    df["recruiter_score"] = (
        4 * df["open_to_work_flag"] +
        4 * df["recruiter_response_rate"] +
        2 * df["profile_views_received_30d"] / 10 +
        2 * df["saved_by_recruiters_30d"] -
        df["notice_period_days"] / 30 -
        df["avg_response_time_hours"] / 100
    )

    return df[["candidate_id", "recruiter_score"]]

def build_trust_score(features):

    df = features.copy()

    df["trust_score"] = (
        df["verified_email"] +
        df["verified_phone"] +
        df["linkedin_connected"] +
        df["profile_completeness_score"] / 20 +
        5 * df["interview_completion_rate"]
    )

    return df[["candidate_id", "trust_score"]]

# master function 

def build_all_features(
    profiles_df,
    career_df,
    skill_df,
    signals_df
):
    """
    Builds the complete feature table.
    """

    features = build_experience_features(profiles_df)

    features = features.merge(
        build_title_features(profiles_df),
        on="candidate_id",
        how="left"
    )

    features = features.merge(
        build_career_stability_features(career_df),
        on="candidate_id",
        how="left"
    )

    features = features.merge(
        build_product_company_features(career_df),
        on="candidate_id",
        how="left"
    )

    features = features.merge(
    build_career_description_features(career_df),
    on="candidate_id",
    how="left"
    )

    features = features.merge(
    build_python_features(skill_df),
    on="candidate_id",
    how="left"
    )

    features = features.merge(
    build_embedding_features(skill_df),
    on="candidate_id",
    how="left"
    )

    features = features.merge(
    build_vector_database_features(skill_df),
    on="candidate_id",
    how="left"
    )

    features = features.merge(
    build_skill_summary(skill_df),
    on="candidate_id",
    how="left"
    )

    features = features.merge(
    build_behavior_features(signals_df),
    on="candidate_id",
    how="left"
    )

    features = features.merge(
    build_ai_skill_score(features),
    on="candidate_id"
)

    features = features.merge(
    build_production_score(features),
    on="candidate_id"
)

    features = features.merge(
    build_career_quality_score(features),
    on="candidate_id"
)

    features = features.merge(
    build_recruiter_score(features),
    on="candidate_id"
)

    features = features.merge(
    build_trust_score(features),
    on="candidate_id"
)

    features = features.fillna(0)

    return features