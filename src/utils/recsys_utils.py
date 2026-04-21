from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import json
import logging
from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer


LOGGER = logging.getLogger("recsys_utils")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


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


def get_clean_cache_dir() -> Path:
    root = get_project_root()
    return ensure_dir(root / "data" / "processed" / "cache" / "clean")


def get_artifact_cache_dir() -> Path:
    root = get_project_root()
    return ensure_dir(root / "data" / "processed" / "cache" / "artifacts")


def load_imdb_table(path: Path) -> pd.DataFrame:
    LOGGER.info("Cargando IMDb: %s", path.name)
    df = pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        na_values="\\N",
        keep_default_na=True,
        low_memory=False,
    )
    LOGGER.info("IMDb cargado: %s | filas=%s columnas=%s", path.name, len(df), df.shape[1])
    return df


def load_movielens_table(path: Path) -> pd.DataFrame:
    LOGGER.info("Cargando MovieLens: %s", path.name)
    df = pd.read_csv(path, low_memory=False)
    LOGGER.info("MovieLens cargado: %s | filas=%s columnas=%s", path.name, len(df), df.shape[1])
    return df


def load_all_tables(
    source: Literal["both", "imdb", "movielens"] = "both",
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    _, imdb_dir, movielens_dir = get_data_dirs()
    LOGGER.info("Buscando datasets | source=%s", source)
    imdb_tables: dict[str, pd.DataFrame] = {}
    movielens_tables: dict[str, pd.DataFrame] = {}

    if source in {"both", "imdb"}:
        LOGGER.info("Explorando carpeta IMDb: %s", imdb_dir)
        imdb_tables = {
            path.stem.replace(".tsv", ""): load_imdb_table(path)
            for path in sorted(imdb_dir.glob("*.tsv.gz"))
        }

    if source in {"both", "movielens"}:
        LOGGER.info("Explorando carpeta MovieLens: %s", movielens_dir)
        movielens_tables = {
            path.stem: load_movielens_table(path)
            for path in sorted(movielens_dir.glob("*.csv"))
        }

    LOGGER.info(
        "Carga completa | tablas IMDb=%s | tablas MovieLens=%s",
        len(imdb_tables),
        len(movielens_tables),
    )
    return imdb_tables, movielens_tables


def _split_pipe_or_comma(value: object, sep: str) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text == "(no genres listed)":
        return []
    return [part for part in text.split(sep) if part]


def clean_imdb_tables(imdb_tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    LOGGER.info("Iniciando limpieza de tablas IMDb")
    imdb = {name: df.copy() for name, df in imdb_tables.items()}

    LOGGER.info("Limpiando IMDb title.basics")
    imdb["title.basics"] = imdb["title.basics"].drop_duplicates(subset="tconst")
    imdb["title.basics"]["isAdult"] = imdb["title.basics"]["isAdult"].astype("Int64")
    for col in ["startYear", "endYear", "runtimeMinutes"]:
        imdb["title.basics"][col] = pd.to_numeric(imdb["title.basics"][col], errors="coerce").astype("Int64")
    imdb["title.basics"]["genres_list"] = imdb["title.basics"]["genres"].apply(lambda x: _split_pipe_or_comma(x, ","))

    LOGGER.info("Limpiando IMDb title.ratings")
    imdb["title.ratings"] = imdb["title.ratings"].drop_duplicates(subset="tconst")
    imdb["title.ratings"]["averageRating"] = pd.to_numeric(imdb["title.ratings"]["averageRating"], errors="coerce")
    imdb["title.ratings"]["numVotes"] = pd.to_numeric(imdb["title.ratings"]["numVotes"], errors="coerce").astype("Int64")

    LOGGER.info("Limpiando IMDb title.crew")
    imdb["title.crew"] = imdb["title.crew"].drop_duplicates(subset="tconst")
    for col in ["directors", "writers"]:
        imdb["title.crew"][f"{col}_list"] = imdb["title.crew"][col].apply(lambda x: _split_pipe_or_comma(x, ","))

    LOGGER.info("Limpiando IMDb title.episode")
    for col in ["seasonNumber", "episodeNumber"]:
        imdb["title.episode"][col] = pd.to_numeric(imdb["title.episode"][col], errors="coerce").astype("Int64")

    LOGGER.info("Limpiando IMDb title.principals")
    imdb["title.principals"]["ordering"] = pd.to_numeric(
        imdb["title.principals"]["ordering"], errors="coerce"
    ).astype("Int64")
    imdb["title.principals"] = imdb["title.principals"].drop_duplicates()

    LOGGER.info("Limpiando IMDb title.akas")
    imdb["title.akas"]["ordering"] = pd.to_numeric(imdb["title.akas"]["ordering"], errors="coerce").astype("Int64")
    imdb["title.akas"]["isOriginalTitle"] = pd.to_numeric(
        imdb["title.akas"]["isOriginalTitle"], errors="coerce"
    ).astype("Int64")
    imdb["title.akas"] = imdb["title.akas"].drop_duplicates()
    for col in ["types", "attributes"]:
        imdb["title.akas"][f"{col}_list"] = imdb["title.akas"][col].apply(lambda x: _split_pipe_or_comma(x, ","))

    LOGGER.info("Limpiando IMDb name.basics")
    imdb["name.basics"] = imdb["name.basics"].drop_duplicates(subset="nconst")
    for col in ["birthYear", "deathYear"]:
        imdb["name.basics"][col] = pd.to_numeric(imdb["name.basics"][col], errors="coerce").astype("Int64")
    for col in ["primaryProfession", "knownForTitles"]:
        imdb["name.basics"][f"{col}_list"] = imdb["name.basics"][col].apply(lambda x: _split_pipe_or_comma(x, ","))

    LOGGER.info("Limpieza IMDb completada")
    return imdb


def clean_movielens_tables(movielens_tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    LOGGER.info("Iniciando limpieza de tablas MovieLens")
    movielens = {name: df.copy() for name, df in movielens_tables.items()}

    LOGGER.info("Limpiando MovieLens movies")
    movielens["movies"] = movielens["movies"].drop_duplicates(subset="movieId")
    movielens["movies"]["movieId"] = pd.to_numeric(movielens["movies"]["movieId"], errors="coerce").astype("Int64")
    movielens["movies"]["genres_list"] = movielens["movies"]["genres"].apply(lambda x: _split_pipe_or_comma(x, "|"))
    movielens["movies"]["release_year"] = movielens["movies"]["title"].str.extract(r"\((\d{4})\)$")[0].astype("Int64")
    movielens["movies"]["clean_title"] = movielens["movies"]["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)

    LOGGER.info("Limpiando MovieLens ratings")
    for col in ["userId", "movieId"]:
        movielens["ratings"][col] = pd.to_numeric(movielens["ratings"][col], errors="coerce").astype("Int64")
    movielens["ratings"]["rating"] = pd.to_numeric(movielens["ratings"]["rating"], errors="coerce")
    movielens["ratings"]["timestamp"] = pd.to_datetime(movielens["ratings"]["timestamp"], unit="s", utc=True)
    movielens["ratings"] = movielens["ratings"].drop_duplicates()

    LOGGER.info("Limpiando MovieLens links")
    for col in ["movieId", "imdbId", "tmdbId"]:
        movielens["links"][col] = pd.to_numeric(movielens["links"][col], errors="coerce").astype("Int64")
    movielens["links"] = movielens["links"].drop_duplicates(subset="movieId")
    movielens["links"]["tconst"] = movielens["links"]["imdbId"].apply(
        lambda x: pd.NA if pd.isna(x) else f"tt{int(x):07d}"
    )

    LOGGER.info("Limpiando MovieLens tags")
    for col in ["userId", "movieId"]:
        movielens["tags"][col] = pd.to_numeric(movielens["tags"][col], errors="coerce").astype("Int64")
    movielens["tags"]["tag"] = movielens["tags"]["tag"].astype("string").str.strip()
    movielens["tags"]["timestamp"] = pd.to_datetime(movielens["tags"]["timestamp"], unit="s", utc=True)
    movielens["tags"] = movielens["tags"].drop_duplicates()

    LOGGER.info("Limpieza MovieLens completada")
    return movielens


def _cache_source_dir(source_name: str) -> Path:
    return ensure_dir(get_clean_cache_dir() / source_name)


def _cache_table_path(source_name: str, table_name: str) -> Path:
    return _cache_source_dir(source_name) / f"{table_name}.pkl"


def _save_clean_tables_to_cache(source_name: str, tables: dict[str, pd.DataFrame]) -> None:
    if not tables:
        return
    LOGGER.info("Guardando cache limpia para %s", source_name)
    for table_name, df in tables.items():
        cache_path = _cache_table_path(source_name, table_name)
        df.to_pickle(cache_path)
        LOGGER.info("Cache guardada: %s", cache_path.relative_to(get_project_root()))


def _load_clean_tables_from_cache(source_name: str) -> dict[str, pd.DataFrame]:
    cache_dir = _cache_source_dir(source_name)
    cache_files = sorted(cache_dir.glob("*.pkl"))
    if not cache_files:
        LOGGER.info("No existe cache limpia para %s", source_name)
        return {}

    LOGGER.info("Cargando cache limpia para %s", source_name)
    tables: dict[str, pd.DataFrame] = {}
    for cache_file in cache_files:
        table_name = cache_file.stem
        tables[table_name] = pd.read_pickle(cache_file)
        LOGGER.info(
            "Cache cargada: %s | filas=%s columnas=%s",
            cache_file.name,
            len(tables[table_name]),
            tables[table_name].shape[1],
        )
    return tables


def load_clean_data(
    source: Literal["both", "imdb", "movielens"] = "both",
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    imdb: dict[str, pd.DataFrame] = {}
    movielens: dict[str, pd.DataFrame] = {}

    LOGGER.info(
        "Iniciando pipeline de carga y limpieza | source=%s | use_cache=%s | refresh_cache=%s",
        source,
        use_cache,
        refresh_cache,
    )

    need_imdb = source in {"both", "imdb"}
    need_movielens = source in {"both", "movielens"}

    if use_cache and not refresh_cache and need_imdb:
        imdb = _load_clean_tables_from_cache("imdb")
    if use_cache and not refresh_cache and need_movielens:
        movielens = _load_clean_tables_from_cache("movielens")

    if need_imdb and not imdb:
        LOGGER.info("No hay cache utilizable de IMDb. Se cargara desde origen.")
        imdb_tables, _ = load_all_tables(source="imdb")
        imdb = clean_imdb_tables(imdb_tables)
        if use_cache:
            _save_clean_tables_to_cache("imdb", imdb)
    elif not need_imdb:
        LOGGER.info("Se omite carga de IMDb porque no fue solicitado")

    if need_movielens and not movielens:
        LOGGER.info("No hay cache utilizable de MovieLens. Se cargara desde origen.")
        _, movielens_tables = load_all_tables(source="movielens")
        movielens = clean_movielens_tables(movielens_tables)
        if use_cache:
            _save_clean_tables_to_cache("movielens", movielens)
    elif not need_movielens:
        LOGGER.info("Se omite carga de MovieLens porque no fue solicitado")

    LOGGER.info("Pipeline de carga y limpieza completado | source=%s", source)
    return imdb, movielens


def summarize_tables(imdb: dict[str, pd.DataFrame], movielens: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for source, tables in [("IMDb", imdb), ("MovieLens", movielens)]:
        for name, df in tables.items():
            rows.append({"source": source, "table": name, "rows": len(df), "columns": df.shape[1]})
    return pd.DataFrame(rows).sort_values(["source", "table"]).reset_index(drop=True)


def build_movies_enriched(
    imdb: dict[str, pd.DataFrame],
    movielens: dict[str, pd.DataFrame],
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_name: str = "movies_enriched",
) -> pd.DataFrame:
    cache_path = get_artifact_cache_dir() / f"{cache_name}.parquet"
    if use_cache and not refresh_cache and cache_path.exists():
        LOGGER.info("Cargando tabla enriquecida desde cache: %s", cache_path.relative_to(get_project_root()))
        movies = pd.read_parquet(cache_path)
        LOGGER.info("Tabla enriquecida cargada desde cache | filas=%s columnas=%s", len(movies), movies.shape[1])
        return movies

    LOGGER.info("Construyendo tabla enriquecida de peliculas")
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
    movies["genres_imdb_list"] = movies["genres_list_imdb"].apply(lambda x: x if isinstance(x, list) else [])
    movies["genres_movielens_list"] = movies["genres_list_movielens"].apply(lambda x: x if isinstance(x, list) else [])
    movies["directors_list"] = movies["directors_list"].apply(lambda x: x if isinstance(x, list) else [])
    movies["writers_list"] = movies["writers_list"].apply(lambda x: x if isinstance(x, list) else [])
    movies["principal_names"] = movies["principal_names"].apply(lambda x: x if isinstance(x, list) else [])
    if use_cache:
        LOGGER.info("Guardando tabla enriquecida en cache: %s", cache_path.relative_to(get_project_root()))
        movies.to_parquet(cache_path, index=False)
    LOGGER.info("Tabla enriquecida construida | filas=%s columnas=%s", len(movies), movies.shape[1])
    return movies


def build_interactions(movielens: dict[str, pd.DataFrame]) -> pd.DataFrame:
    interactions = movielens["ratings"][["userId", "movieId", "rating", "timestamp"]].copy()
    LOGGER.info("Interacciones preparadas | filas=%s", len(interactions))
    return interactions


def filter_min_interactions(
    ratings: pd.DataFrame,
    min_user_ratings: int = 20,
    min_item_ratings: int = 20,
    iterations: int = 3,
) -> pd.DataFrame:
    LOGGER.info(
        "Filtrando interacciones | min_user_ratings=%s | min_item_ratings=%s | iterations=%s",
        min_user_ratings,
        min_item_ratings,
        iterations,
    )
    filtered = ratings.copy()
    for iteration in range(iterations):
        user_counts = filtered.groupby("userId").size()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        filtered = filtered[filtered["userId"].isin(valid_users)]

        item_counts = filtered.groupby("movieId").size()
        valid_items = item_counts[item_counts >= min_item_ratings].index
        filtered = filtered[filtered["movieId"].isin(valid_items)]
        LOGGER.info(
            "Filtro iteracion %s/%s | filas=%s usuarios=%s items=%s",
            iteration + 1,
            iterations,
            len(filtered),
            filtered["userId"].nunique(),
            filtered["movieId"].nunique(),
        )
    return filtered.reset_index(drop=True)


def temporal_split(
    ratings: pd.DataFrame,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    LOGGER.info(
        "Generando split temporal | train_ratio=%.2f | valid_ratio=%.2f",
        train_ratio,
        valid_ratio,
    )
    ratings = ratings.sort_values("timestamp").reset_index(drop=True)
    n_rows = len(ratings)
    train_end = int(n_rows * train_ratio)
    valid_end = int(n_rows * (train_ratio + valid_ratio))
    train = ratings.iloc[:train_end].copy()
    valid = ratings.iloc[train_end:valid_end].copy()
    test = ratings.iloc[valid_end:].copy()
    LOGGER.info("Split temporal listo | train=%s valid=%s test=%s", len(train), len(valid), len(test))
    return train, valid, test


def create_index_maps(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    LOGGER.info("Creando mapas de indices de usuarios e items")
    user_index = pd.DataFrame({"userId": sorted(ratings["userId"].dropna().unique())})
    user_index["user_idx"] = np.arange(len(user_index), dtype=np.int64)

    item_index = pd.DataFrame({"movieId": sorted(ratings["movieId"].dropna().unique())})
    item_index["item_idx"] = np.arange(len(item_index), dtype=np.int64)
    LOGGER.info("Mapas creados | users=%s | items=%s", len(user_index), len(item_index))
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
    if text_per_item.empty or text_per_item["tag"].fillna("").str.strip().eq("").all():
        LOGGER.info("No hay texto suficiente en tags para construir TF-IDF")
        empty_features = text_per_item[["movieId"]].copy()
        return empty_features, TfidfVectorizer()

    vectorizer = TfidfVectorizer(min_df=min_df, max_features=max_features, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(text_per_item["tag"].fillna(""))
    feature_df = pd.DataFrame.sparse.from_spmatrix(
        matrix,
        index=text_per_item["movieId"],
        columns=[f"tag_tfidf_{name}" for name in vectorizer.get_feature_names_out()],
    )
    return feature_df.reset_index(), vectorizer


def densify_sparse_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    sparse_cols = [col for col in output.columns if isinstance(output[col].dtype, pd.SparseDtype)]
    if sparse_cols:
        LOGGER.info("Convirtiendo columnas sparse a densas | n_cols=%s", len(sparse_cols))
    for col in sparse_cols:
        output[col] = output[col].sparse.to_dense().astype("float32")
    return output


def prepare_content_features(items: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    LOGGER.info("Preparando features de contenido")
    frame = items[["movieId", "title", "clean_title", "release_year", "startYear", "runtimeMinutes", "averageRating", "numVotes"]].copy()
    frame["release_year"] = frame["release_year"].fillna(frame["startYear"])

    LOGGER.info("Construyendo matrices de generos")
    genre_ml = build_genre_matrix(items, "genres_movielens_list")
    genre_imdb = build_genre_matrix(items, "genres_imdb_list")
    feature_frame = pd.concat([frame.reset_index(drop=True), genre_ml.reset_index(drop=True), genre_imdb.reset_index(drop=True)], axis=1)

    LOGGER.info("Construyendo features TF-IDF de tags")
    tag_features, _ = build_tag_features(tags)
    feature_frame = feature_frame.merge(tag_features, on="movieId", how="left")
    feature_frame = densify_sparse_columns(feature_frame)
    LOGGER.info("Features de contenido listas | filas=%s columnas=%s", len(feature_frame), feature_frame.shape[1])
    return feature_frame


def build_item_knn_model(
    interactions: pd.DataFrame,
    n_users: int,
    n_items: int,
    n_neighbors: int = 50,
) -> tuple[NearestNeighbors, sparse.csr_matrix]:
    LOGGER.info(
        "Entrenando modelo item-kNN | filas_interacciones=%s | users=%s | items=%s | neighbors=%s",
        len(interactions),
        n_users,
        n_items,
        n_neighbors,
    )
    matrix = build_sparse_interaction_matrix(interactions, n_users=n_users, n_items=n_items).T.tocsr()
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
    model.fit(matrix)
    LOGGER.info("Modelo item-kNN listo")
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
