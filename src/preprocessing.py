import pandas as pd

def build_profiles_df(candidates):
    """
    Creates a DataFrame containing profile information for every candidate.
    """

    profiles = []

    for candidate in candidates:

        profile = candidate["profile"]

        profiles.append({
            "candidate_id": candidate["candidate_id"],
            **profile
        })

    return pd.DataFrame(profiles)

def build_skill_df(candidates):
    """
    Creates a row for every candidate skill.
    """

    skill_rows = []

    for candidate in candidates:

        for skill in candidate["skills"]:

            skill_rows.append({

                "candidate_id": candidate["candidate_id"],

                "skill_name": skill["name"],

                "proficiency": skill["proficiency"],

                "endorsements": skill["endorsements"],

                "duration_months": skill.get("duration_months", 0)

            })

    return pd.DataFrame(skill_rows)

def build_career_df(candidates):
    """
    Creates one row per job in a candidate's career history.
    """

    career_rows = []

    for candidate in candidates:

        candidate_id = candidate["candidate_id"]

        for job in candidate["career_history"]:

            career_rows.append({

                "candidate_id": candidate_id,

                "company": job["company"],

                "title": job["title"],

                "industry": job["industry"],

                "company_size": job["company_size"],

                "start_date": job["start_date"],

                "end_date": job["end_date"],

                "duration_months": job["duration_months"],

                "is_current": job["is_current"],

                "description": job["description"]

            })

    return pd.DataFrame(career_rows)

def build_signals_df(candidates):
    """
    Creates one row per candidate containing all behavioral signals.
    """

    signal_rows = []

    for candidate in candidates:

        signals = candidate["redrob_signals"]

        signal_rows.append({

            "candidate_id": candidate["candidate_id"],

            **signals

        })

    return pd.DataFrame(signal_rows)

def prepare_data(candidates):
    """
    Builds every DataFrame required throughout the project.

    Returns
    -------
    dict
        {
            "profiles": DataFrame,
            "skills": DataFrame,
            "career": DataFrame,
            "signals": DataFrame
        }
    """

    return {

        "profiles": build_profiles_df(candidates),

        "skills": build_skill_df(candidates),

        "career": build_career_df(candidates),

        "signals": build_signals_df(candidates)

    }