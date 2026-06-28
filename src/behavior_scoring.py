import pandas as pd
import numpy as np


def min_max_scale(series):
    """
    Min-Max normalize a numeric series to [0,1].
    """

    series = series.fillna(0)

    if series.max() == series.min():
        return pd.Series(0.0, index=series.index)

    return (series - series.min()) / (series.max() - series.min())



# Availability


def build_availability_score(df):

    data = df.copy()

    data["open_to_work"] = data["open_to_work_flag"].astype(int)

    response_rate = data["recruiter_response_rate"].clip(0, 1)

    response_time = 1 - min_max_scale(data["avg_response_time_hours"])

    notice = 1 - min_max_scale(data["notice_period_days"])

    availability = (
        0.35 * data["open_to_work"] +
        0.30 * response_rate +
        0.20 * response_time +
        0.15 * notice
    )

    return pd.DataFrame({
        "candidate_id": data["candidate_id"],
        "availability_score": availability
    })



# Recruiter Interest


def build_recruiter_interest_score(df):

    data = df.copy()

    profile_views = min_max_scale(data["profile_views_received_30d"])

    search = min_max_scale(data["search_appearance_30d"])

    saved = min_max_scale(data["saved_by_recruiters_30d"])

    endorsements = min_max_scale(data["endorsements_received"])

    connections = min_max_scale(data["connection_count"])

    recruiter_interest = (
        0.25 * profile_views +
        0.25 * search +
        0.20 * saved +
        0.15 * endorsements +
        0.15 * connections
    )

    return pd.DataFrame({
        "candidate_id": data["candidate_id"],
        "recruiter_interest_score": recruiter_interest
    })



# Trust


def build_trust_score(df):

    data = df.copy()

    data["verified_email"] = data["verified_email"].astype(int)

    data["verified_phone"] = data["verified_phone"].astype(int)

    data["linkedin_connected"] = data["linkedin_connected"].astype(int)

    completeness = data["profile_completeness_score"] / 100

    trust = (
        0.40 * completeness +
        0.20 * data["verified_email"] +
        0.20 * data["verified_phone"] +
        0.20 * data["linkedin_connected"]
    )

    return pd.DataFrame({
        "candidate_id": data["candidate_id"],
        "trust_score": trust
    })



# Engagement


def build_engagement_score(df):

    data = df.copy()

    github = data["github_activity_score"].replace(-1, 0) / 100

    github = github.clip(0, 1)

    interview = data["interview_completion_rate"].clip(0, 1)

    offer = data["offer_acceptance_rate"].replace(-1, 0).clip(0, 1)

    assessments = data["skill_assessment_scores"].apply(
        lambda x: len(x) if isinstance(x, dict) else 0
    )

    assessments = min_max_scale(assessments)

    engagement = (
        0.35 * interview +
        0.30 * offer +
        0.20 * github +
        0.15 * assessments
    )

    return pd.DataFrame({
        "candidate_id": data["candidate_id"],
        "engagement_score": engagement
    })


# master function

def build_behavior_score(signals_df):

    availability = build_availability_score(signals_df)

    recruiter = build_recruiter_interest_score(signals_df)

    trust = build_trust_score(signals_df)

    engagement = build_engagement_score(signals_df)

    behavior = (
        availability
        .merge(recruiter, on="candidate_id")
        .merge(trust, on="candidate_id")
        .merge(engagement, on="candidate_id")
    )

    behavior["behavior_score"] = (
    0.40 * behavior["availability_score"] +
    0.15 * behavior["recruiter_interest_score"] +
    0.20 * behavior["trust_score"] +
    0.25 * behavior["engagement_score"]
)

    return behavior