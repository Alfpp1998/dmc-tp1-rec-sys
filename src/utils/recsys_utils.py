from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer


def get_project_root() -> Path:
    cwd = Path.cwd().resolve()
    if cwd.name in {"notebooks", "test"}:
        return cwd.parent
    return cwd


def get_data_dirs() -> tuple[Path, Path, Path]:
    root = get_project_root()
    data_dir = root / "data"
    return root, data_dir / "imdb", data_dir / "mov_lens"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_imdb_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        na_values="\\N",
        keep_default_na=True,
        low_memory=False,
    )


def load_movielens_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def load_all_tables() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    _, imdb_dir, movielens_dir = get_data_dirs()
    imdb_tables = {
        path.stem.replace(".tsv", ""): load_imdb_table(path)
        for path in sorted(imdb_dir.glob("*.tsv.gz"))
    }
    movielens_tables = {
        path.stem: load_movielens_table(path)
        for path in sorted(movielens_dir.glob("*.csv"))
    }
    return imdb_tables, movielens_tables


def _split_pipe_or_comma(value: object, sep: str) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text == "(no genres listed)":
        return []
    return [part for part in text.split(sep) if part]


def clean_imdb_tables(imdb_tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    imdb = {name: df.copy() for name, df in imdb_tables.items()}

    imdb["title.basics"] = imdb["title.basics"].drop_duplicates(subset="tconst")
    imdb["title.basics"]["isAdult"] = imdb["title.basics"]["isAdult"].astype("Int64")
    for col in ["startYear", "endYear", "runtimeMinutes"]:
        imdb["title.basics"][col] = pd.to_numeric(imdb["title.basics"][col], errors="coerce").astype("Int64")
    imdb["title.basics"]["genres_list"] = imdb["title.basics"]["genres"].apply(lambda x: _split_pipe_or_comma(x, ","))

    imdb["title.ratings"] = imdb["title.ratings"].drop_duplicates(subset="tconst")
    imdb["title.ratings"]["averageRating"] = pd.to_numeric(imdb["title.ratings"]["averageRating"], errors="coerce")
    imdb["title.ratings"]["numVotes"] = pd.to_numeric(imdb["title.ratings"]["numVotes"], errors="coerce").astype("Int64")

    imdb["title.crew"] = imdb["title.crew"].drop_duplicates(subset="tconst")
    for col in ["directors", "writers"]:
        imdb["title.crew"][f"{col}_list"] = imdb["title.crew"][col].apply(lambda x: _split_pipe_or_comma(x, ","))

    for col in ["seasonNumber", "episodeNumber"]:
        imdb["title.episode"][col] = pd.to_numeric(imdb["title.episode"][col], errors="coerce").astype("Int64")

    imdb["title.principals"]["ordering"] = pd.to_numeric(
        imdb["title.principals"]["ordering"], errors="coerce"
    ).astype("Int64")
    imdb["title.principals"] = imdb["title.principals"].drop_duplicates()

    imdb["title.akas"]["ordering"] = pd.to_numeric(imdb["title.akas"]["ordering"], errors="coerce").astype("Int64")
    imdb["title.akas"]["isOriginalTitle"] = pd.to_numeric(
        imdb["title.akas"]["isOriginalTitle"], errors="coerce"
    ).astype("Int64")
    imdb["title.akas"] = imdb["title.akas"].drop_duplicates()
    for col in ["types", "attributes"]:
        imdb["title.akas"][f"{col}_list"] = imdb["title.akas"][col].apply(lambda x: _split_pipe_or_comma(x, ","))

    imdb["name.basics"] = imdb["name.basics"].drop_duplicates(subset="nconst")
    for col in ["birthYear", "deathYear"]:
        imdb["name.basics"][col] = pd.to_numeric(imdb["name.basics"][col], errors="coerce").astype("Int64")
    for col in ["primaryProfession", "knownForTitles"]:
        imdb["name.basics"][f"{col}_list"] = imdb["name.basics"][col].apply(lambda x: _split_pipe_or_comma(x, ","))

    return imdb


def clean_movielens_tables(movielens_tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    movielens = {name: df.copy() for name, df in movielens_tables.items()}

    movielens["movies"] = movielens["movies"].drop_duplicates(subset="movieId")
    movielens["movies"]["movieId"] = pd.to_numeric(movielens["movies"]["movieId"], errors="coerce").astype("Int64")
    movielens["movies"]["genres_list"] = movielens["movies"]["genres"].apply(lambda x: _split_pipe_or_comma(x, "|"))
    movielens["movies"]["release_year"] = movielens["movies"]["title"].str.extract(r"\((\d{4})\)$")[0].astype("Int64")
    movielens["movies"]["clean_title"] = movielens["movies"]["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)

    for col in ["userId", "movieId"]:
        movielens["ratings"][col] = pd.to_numeric(movielens["ratings"][col], errors="coerce").astype("Int64")
    movielens["ratings"]["rating"] = pd.to_numeric(movielens["ratings"]["rating"], errors="coerce")
    movielens["ratings"]["timestamp"] = pd.to_datetime(movielens["ratings"]["timestamp"], unit="s", utc=True)
    movielens["ratings"] = movielens["ratings"].drop_duplicates()

    for col in ["movieId", "imdbId", "tmdbId"]:
        movielens["links"][col] = pd.to_numeric(movielens["links"][col], errors="coerce").astype("Int64")
    movielens["links"] = movielens["links"].drop_duplicates(subset="movieId")
    movielens["links"]["tconst"] = movielens["links"]["imdbId"].apply(
        lambda x: pd.NA if pd.isna(x) else f"tt{int(x):07d}"
    )

    for col in ["userId", "movieId"]:
        movielens["tags"][col] = pd.to_numeric(movielens["tags"][col], errors="coerce").astype("Int64")
    movielens["tags"]["tag"] = movielens["tags"]["tag"].astype("string").str.strip()
    movielens["tags"]["timestamp"] = pd.to_datetime(movielens["tags"]["timestamp"], unit="s", utc=True)
    movielens["tags"] = movielens["tags"].drop_duplicates()

    return movielens


def load_clean_data() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    imdb_tables, movielens_tables = load_all_tables()
    return clean_imdb_tables(imdb_tables), clean_movielens_tables(movielens_tables)


def summarize_tables(imdb: dict[str, pd.DataFrame], movielens: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for source, tables in [("IMDb", imdb), ("MovieLens", movielens)]:
        for name, df in tables.items():
            rows.append({"source": source, "table": name, "rows": len(df), "columns": df.shape[1]})
    return pd.DataFrame(rows).sort_values(["source", "table"]).reset_index(drop=True)


def build_movies_enriched(imdb: dict[str, pd.DataFrame], movielens: dict[str, pd.DataFrame]) -> pd.DataFrame:
    crew_cols = imdb["title.crew"][["tconst", "directors_list", "writers_list"]]
    principal_names = (
        imdb["title.principals"]
        .merge(imdb["name.basics"][["nconst", "primaryName"]], on="nconst", how="left")
        .sort_values(["tconst", "ordering"])
        .groupby("tconst")["primaryName"]
        .apply(lambda names: [name for name in names.dropna().tolist()[:5]])
        .rename("principal_names")
        .reset_index()
    )

    movies = (
        movielens["movies"]
        .merge(movielens["links"][["movieId", "tconst", "tmdbId"]], on="movieId", how="left")
        .merge(
            imdb["title.basics"][
                [
                    "tconst",
                    "titleType",
                    "primaryTitle",
                    "originalTitle",
                    "isAdult",
                    "startYear",
                    "runtimeMinutes",
                    "genres_list",
                ]
            ],
            on="tconst",
            how="left",
            suffixes=("_movielens", "_imdb"),
        )
        .merge(imdb["title.ratings"][["tconst", "averageRating", "numVotes"]], on="tconst", how="left")
        .merge(crew_cols, on="tconst", how="left")
        .merge(principal_names, on="tconst", how="left")
    )
    movies["genres_imdb_list"] = movies["genres_list"].apply(lambda x: x if isinstance(x, list) else [])
    movies["genres_movielens_list"] = movies["genres_list_movielens"].apply(lambda x: x if isinstance(x, list) else [])
    movies["directors_list"] = movies["directors_list"].apply(lambda x: x if isinstance(x, list) else [])
    movies["writers_list"] = movies["writers_list"].apply(lambda x: x if isinstance(x, list) else [])
    movies["principal_names"] = movies["principal_names"].apply(lambda x: x if isinstance(x, list) else [])
    return movies


def build_interactions(movielens: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return movielens["ratings"][["userId", "movieId", "rating", "timestamp"]].copy()


def filter_min_interactions(
    ratings: pd.DataFrame,
    min_user_ratings: int = 20,
    min_item_ratings: int = 20,
    iterations: int = 3,
) -> pd.DataFrame:
    filtered = ratings.copy()
    for _ in range(iterations):
        user_counts = filtered.groupby("userId").size()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        filtered = filtered[filtered["userId"].isin(valid_users)]

        item_counts = filtered.groupby("movieId").size()
        valid_items = item_counts[item_counts >= min_item_ratings].index
        filtered = filtered[filtered["movieId"].isin(valid_items)]
    return filtered.reset_index(drop=True)


def temporal_split(
    ratings: pd.DataFrame,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings = ratings.sort_values("timestamp").reset_index(drop=True)
    n_rows = len(ratings)
    train_end = int(n_rows * train_ratio)
    valid_end = int(n_rows * (train_ratio + valid_ratio))
    train = ratings.iloc[:train_end].copy()
    valid = ratings.iloc[train_end:valid_end].copy()
    test = ratings.iloc[valid_end:].copy()
    return train, valid, test


def create_index_maps(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_index = pd.DataFrame({"userId": sorted(ratings["userId"].dropna().unique())})
    user_index["user_idx"] = np.arange(len(user_index), dtype=np.int64)

    item_index = pd.DataFrame({"movieId": sorted(ratings["movieId"].dropna().unique())})
    item_index["item_idx"] = np.arange(len(item_index), dtype=np.int64)
    return user_index, item_index


def add_indices(ratings: pd.DataFrame, user_index: pd.DataFrame, item_index: pd.DataFrame) -> pd.DataFrame:
    return (
        ratings.merge(user_index, on="userId", how="inner")
        .merge(item_index, on="movieId", how="inner")
        .sort_values(["timestamp", "user_idx", "item_idx"])
        .reset_index(drop=True)
    )


def build_sparse_interaction_matrix(
    ratings: pd.DataFrame,
    n_users: int,
    n_items: int,
    value_col: str = "rating",
) -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (ratings[value_col].astype(float), (ratings["user_idx"], ratings["item_idx"])),
        shape=(n_users, n_items),
    )


def popularity_ranking(train_ratings: pd.DataFrame) -> pd.DataFrame:
    ranking = (
        train_ratings.groupby("movieId")
        .agg(popularity_score=("rating", "mean"), rating_count=("rating", "size"))
        .reset_index()
        .sort_values(["rating_count", "popularity_score"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return ranking


def build_genre_matrix(items: pd.DataFrame, genre_col: str) -> pd.DataFrame:
    mlb = MultiLabelBinarizer()
    transformed = mlb.fit_transform(items[genre_col].apply(lambda x: x if isinstance(x, list) else []))
    return pd.DataFrame(transformed, index=items.index, columns=[f"{genre_col}_{label}" for label in mlb.classes_])


def build_tag_features(tags: pd.DataFrame, min_df: int = 20, max_features: int = 500) -> tuple[pd.DataFrame, TfidfVectorizer]:
    text_per_item = tags.groupby("movieId")["tag"].apply(lambda values: " ".join(values.dropna().astype(str))).reset_index()
    vectorizer = TfidfVectorizer(min_df=min_df, max_features=max_features, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(text_per_item["tag"].fillna(""))
    feature_df = pd.DataFrame.sparse.from_spmatrix(
        matrix,
        index=text_per_item["movieId"],
        columns=[f"tag_tfidf_{name}" for name in vectorizer.get_feature_names_out()],
    )
    return feature_df.reset_index(), vectorizer


def prepare_content_features(items: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    frame = items[["movieId", "title", "clean_title", "release_year", "startYear", "runtimeMinutes", "averageRating", "numVotes"]].copy()
    frame["release_year"] = frame["release_year"].fillna(frame["startYear"])

    genre_ml = build_genre_matrix(items, "genres_movielens_list")
    genre_imdb = build_genre_matrix(items, "genres_imdb_list")
    feature_frame = pd.concat([frame.reset_index(drop=True), genre_ml.reset_index(drop=True), genre_imdb.reset_index(drop=True)], axis=1)

    tag_features, _ = build_tag_features(tags)
    feature_frame = feature_frame.merge(tag_features, on="movieId", how="left")
    return feature_frame


def build_item_knn_model(
    interactions: pd.DataFrame,
    n_users: int,
    n_items: int,
    n_neighbors: int = 50,
) -> tuple[NearestNeighbors, sparse.csr_matrix]:
    matrix = build_sparse_interaction_matrix(interactions, n_users=n_users, n_items=n_items).T.tocsr()
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
    model.fit(matrix)
    return model, matrix


def get_top_neighbors(model: NearestNeighbors, item_matrix: sparse.csr_matrix, item_idx: int, n_neighbors: int = 10) -> pd.DataFrame:
    distances, indices = model.kneighbors(item_matrix[item_idx], n_neighbors=n_neighbors + 1)
    pairs = [
        {"neighbor_item_idx": int(idx), "distance": float(dist)}
        for idx, dist in zip(indices[0], distances[0], strict=False)
        if int(idx) != int(item_idx)
    ]
    return pd.DataFrame(pairs)


def make_binary_ground_truth(test_ratings: pd.DataFrame, threshold: float = 4.0) -> dict[int, set[int]]:
    relevant = test_ratings.loc[test_ratings["rating"] >= threshold, ["user_idx", "item_idx"]]
    return relevant.groupby("user_idx")["item_idx"].apply(set).to_dict()


def precision_recall_ndcg_at_k(
    recommendations: pd.DataFrame,
    ground_truth: dict[int, set[int]],
    k: int = 10,
) -> dict[str, float]:
    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []

    for user_idx, user_recs in recommendations.groupby("user_idx"):
        truth = ground_truth.get(int(user_idx), set())
        if not truth:
            continue
        ranked_items = user_recs.sort_values("score", ascending=False)["item_idx"].head(k).tolist()
        hits = [1 if item in truth else 0 for item in ranked_items]
        precisions.append(sum(hits) / k)
        recalls.append(sum(hits) / len(truth))

        dcg = sum(hit / np.log2(rank + 2) for rank, hit in enumerate(hits))
        ideal_hits = [1] * min(len(truth), k)
        idcg = sum(hit / np.log2(rank + 2) for rank, hit in enumerate(ideal_hits))
        ndcgs.append(float(dcg / idcg) if idcg else 0.0)

    return {
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "evaluated_users": int(len(precisions)),
    }


def save_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def sample_recent_interactions(ratings: pd.DataFrame, n_rows: int = 500_000) -> pd.DataFrame:
    return ratings.sort_values("timestamp").tail(n_rows).reset_index(drop=True)


def flatten_column(values: Iterable[list[str]]) -> list[str]:
    flattened: list[str] = []
    for value in values:
        if isinstance(value, list):
            flattened.extend(value)
    return flattened
