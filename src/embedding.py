from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def load_embedding_model(model_name="BAAI/bge-small-en-v1.5"):
    """
    Loads the embedding model.
    """
    return SentenceTransformer(model_name)


def build_candidate_documents(candidates):
    """
    Builds one semantic document per candidate.
    """

    documents = []

    for candidate in candidates:

        profile = candidate["profile"]

        skills = ", ".join(
            skill["name"]
            for skill in candidate["skills"]
        )

        career = " ".join(
            job["description"]
            for job in candidate["career_history"]
        )

        text = f"""
Headline:
{profile["headline"]}

Summary:
{profile["summary"]}

Current Title:
{profile["current_title"]}

Years of Experience:
{profile["years_of_experience"]}

Skills:
{skills}

Career History:
{career}
"""

        documents.append({
            "candidate_id": candidate["candidate_id"],
            "text": text
        })

    return pd.DataFrame(documents)


import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def compute_semantic_scores(
    model_name,
    job_description,
    documents,
    batch_size=512,
    save_path="../data/semantic_scores.parquet"
):
    """
    Computes semantic similarity with automatic checkpointing and resume support.
    """

    model = load_embedding_model(model_name)

    jd_embedding = model.encode(
        job_description,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Resume if previous file exists
    if os.path.exists(save_path):

        semantic_scores = pd.read_parquet(save_path)

        processed = set(semantic_scores["candidate_id"])

        documents = documents[
            ~documents["candidate_id"].isin(processed)
        ].reset_index(drop=True)

        print(f"Resuming... {len(processed)} candidates already processed.")

    else:

        semantic_scores = pd.DataFrame(
            columns=[
                "candidate_id",
                "semantic_similarity"
            ]
        )

    total = len(documents)

    for start in range(0, total, batch_size):

        end = min(start + batch_size, total)

        batch = documents.iloc[start:end]

        embeddings = model.encode(
            batch["text"].tolist(),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        similarity = cosine_similarity(
            embeddings,
            jd_embedding.reshape(1, -1)
        ).flatten()

        batch_scores = pd.DataFrame({
            "candidate_id": batch["candidate_id"].values,
            "semantic_similarity": similarity
        })

        semantic_scores = pd.concat(
            [semantic_scores, batch_scores],
            ignore_index=True
        )

        # Save after every batch
        semantic_scores.to_parquet(
            save_path,
            index=False
        )

        print(
            f"Processed {end}/{total} "
            f"(Saved {len(semantic_scores)} candidates)"
        )

    return semantic_scores