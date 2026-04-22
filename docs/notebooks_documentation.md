# Documentacion de Notebooks y Estructura de Datos

## 1. Proposito del documento

Este documento describe la estructura actual del proyecto, los datasets utilizados, la organizacion esperada de carpetas y la funcion de cada notebook implementado en `notebooks/`.

## 2. Descarga de datasets

Los datasets no se subirán al repositorio. Deben descargarse manualmente desde las fuentes oficiales y luego colocarse en la estructura local ya definida en este proyecto.

### IMDb

Fuente oficial:

- <https://datasets.imdbws.com/>

Archivos utilizados por el proyecto:

- `title.basics.tsv.gz`
- `title.ratings.tsv.gz`
- `title.crew.tsv.gz`
- `title.principals.tsv.gz`
- `name.basics.tsv.gz`
- `title.akas.tsv.gz`
- `title.episode.tsv.gz`

### MovieLens 32M

Fuente oficial:

- <https://grouplens.org/datasets/movielens/32m/>

Archivos utilizados por el proyecto:

- `ratings.csv`
- `movies.csv`
- `tags.csv`
- `links.csv`

## 3. Estructura esperada de carpetas para los datasets

Después de descargar los archivos, deben colocarse localmente siguiendo esta estructura:

```text
dmc-tp1-rec-sys/
├── data/
│   ├── imdb/
│   │   ├── title.basics.tsv.gz
│   │   ├── title.ratings.tsv.gz
│   │   ├── title.crew.tsv.gz
│   │   ├── title.principals.tsv.gz
│   │   ├── name.basics.tsv.gz
│   │   ├── title.akas.tsv.gz
│   │   └── title.episode.tsv.gz
│   └── mov_lens/
│       ├── ratings.csv
│       ├── movies.csv
│       ├── tags.csv
│       └── links.csv
```

Nota:

- este documento no implica borrar ni mover los archivos locales ya existentes
- la carga del proyecto asume exactamente esas carpetas: `data/imdb/` y `data/mov_lens/`

## 4. Estructura actual del repositorio

```text
dmc-tp1-rec-sys/
├── data/
│   ├── imdb/
│   ├── mov_lens/
│   └── processed/
│       └── cache/
├── docs/
│   └── notebooks_documentation.md
├── models/
├── notebooks/
│   ├── 01_eda_ratings_and_catalog.ipynb
│   ├── 02_eda_content_enrichment_imdb.ipynb
│   ├── 03_preprocessing_interactions.ipynb
│   ├── 04_preprocessing_content_features.ipynb
│   ├── 05a_model_cf_item_user.ipynb
│   ├── 05b_model_bandit_offline.ipynb
│   ├── 05c_model_hybrid.ipynb
│   └── 06_evaluation_and_error_analysis.ipynb
├── reports/
│   ├── figures/
│   ├── tables/
│   └── presentation/
├── src/
│   └── utils/
│       └── recsys_utils.py
├── test/
│   └── preprocessing.ipynb
├── README.md
└── requirements.txt
```

## 5. Entorno minimo

El proyecto utiliza el archivo `requirements.txt` para instalar las dependencias minimas.

Inicializacion sugerida del entorno:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Paquetes incluidos:

- `numpy`
- `pandas`
- `pyarrow`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `jupyterlab`
- `ipykernel`
- `tqdm`

## 6. Utilidades compartidas

El archivo [src/utils/recsys_utils.py](/Users/chperezpelaez/Documents/Github/dmc-tp1-rec-sys/src/utils/recsys_utils.py) centraliza la logica compartida del proyecto.

Funciones principales documentadas en el flujo actual:

- carga de tablas IMDb y MovieLens
- limpieza y tipado de columnas
- construccion del join enriquecido entre MovieLens e IMDb
- preparacion de interacciones
- splits temporales
- construccion de features de contenido
- entrenamiento base de item-kNN
- metricas de evaluacion

También se incluyeron logs visibles en consola para que la carga y limpieza de tablas grandes muestre progreso.

## 7. Cache y artefactos intermedios

El proyecto usa cache local para evitar repetir tareas pesadas en cada notebook.

### Cache de tablas limpias

Ruta:

- `data/processed/cache/clean/imdb/`
- `data/processed/cache/clean/movielens/`

Contenido:

- tablas limpias persistidas como `.pkl`

Uso:

- `load_clean_data(..., use_cache=True)` reutiliza estos archivos si ya existen
- si no existen, carga desde origen, limpia y luego los guarda

### Cache de tabla enriquecida

Ruta:

- `data/processed/cache/artifacts/movies_enriched.parquet`

Contenido:

- resultado del join enriquecido entre MovieLens e IMDb

Uso:

- `build_movies_enriched(..., use_cache=True)` reutiliza ese parquet cuando ya fue generado

## 8. Notebooks implementados

### `01_eda_ratings_and_catalog.ipynb`

Funcion:

- análisis exploratorio sobre ratings, catálogo base, tags y links de MovieLens

Inputs:

- `data/mov_lens/ratings.csv`
- `data/mov_lens/movies.csv`
- `data/mov_lens/tags.csv`
- `data/mov_lens/links.csv`

Dependencia de utilidades:

- `load_clean_data(source="movielens", use_cache=True)`

Outputs principales:

- tablas resumen en `reports/tables/`
- figuras exploratorias en `reports/figures/`

Artefactos esperados:

- `reports/tables/eda_table_inventory.csv`
- `reports/tables/eda_catalog_summary.csv`
- `reports/tables/eda_interaction_metrics.csv`
- `reports/tables/eda_catalog_coverage.csv`
- `reports/tables/eda_findings.csv`
- `reports/figures/eda_ratings_users_items.png`
- `reports/figures/eda_genres_tags_timeline.png`

### `02_eda_content_enrichment_imdb.ipynb`

Funcion:

- análisis exploratorio del enriquecimiento de contenido con IMDb y de la cobertura del join MovieLens-IMDb

Inputs:

- tablas de `data/imdb/`
- tablas de `data/mov_lens/`

Dependencia de utilidades:

- `load_clean_data(source="both", use_cache=True)`
- `build_movies_enriched(imdb, movielens, use_cache=True)`

Outputs principales:

- cobertura del join
- distribucion de tipos, duracion y año
- análisis de géneros y señales de personas

Artefactos esperados:

- `reports/tables/imdb_join_coverage.csv`
- `reports/tables/imdb_feature_quality.csv`
- `reports/tables/hybrid_feature_selection.csv`
- `reports/figures/eda_imdb_type_runtime_year.png`
- `reports/figures/eda_imdb_people_and_genres.png`

### `03_preprocessing_interactions.ipynb`

Funcion:

- preparacion de interacciones para entrenamiento y evaluacion

Inputs:

- `data/mov_lens/ratings.csv`
- `data/mov_lens/movies.csv`

Dependencia de utilidades:

- `load_clean_data(source="movielens", use_cache=True)`

Procesamiento documentado:

- construccion de interacciones limpias
- filtrado minimo por actividad
- split temporal
- mapeos `userId -> user_idx`
- mapeos `movieId -> item_idx`
- baseline de popularidad

Artefactos esperados:

- `data/processed/interactions_train.parquet`
- `data/processed/interactions_valid.parquet`
- `data/processed/interactions_test.parquet`
- `data/processed/user_index.parquet`
- `data/processed/item_index.parquet`
- `data/processed/popularity_baseline.parquet`
- `reports/tables/interaction_preprocessing_summary.csv`
- `reports/tables/interaction_split_summary.csv`

### `04_preprocessing_content_features.ipynb`

Funcion:

- construccion de tabla maestra de items y de features de contenido para modelos hibridos y para deep learning

Inputs:

- tablas MovieLens
- tablas IMDb
- join enriquecido entre ambas fuentes

Dependencia de utilidades:

- `load_clean_data(source="both", use_cache=True)`
- `build_movies_enriched(imdb, movielens, use_cache=True)`
- `prepare_content_features(items_master, movielens["tags"])`

Procesamiento documentado:

- construccion de `items_master`
- variables numericas
- codificacion de géneros
- codificacion de `titleType`
- representacion de tags con TF-IDF
- features de directores y principals frecuentes

Artefactos esperados:

- `data/processed/items_master.pkl`
- `data/processed/item_features.parquet`
- `reports/tables/item_feature_quality.csv`

### `05a_model_cf_item_user.ipynb`

Funcion:

- entrenamiento de un baseline colaborativo item-based con item-kNN

Inputs:

- `data/processed/interactions_train.parquet`
- `data/processed/interactions_valid.parquet`
- `data/processed/interactions_test.parquet`
- `data/processed/item_index.parquet`

Procesamiento documentado:

- reduccion a un subconjunto denso para prototipado
- remapeo local de usuarios e items
- entrenamiento item-kNN
- generacion de recomendaciones top-N
- evaluacion sobre test

Artefactos esperados:

- `models/cf_recommendations.parquet`
- `reports/tables/cf_metrics.csv`

### `05b_model_bandit_offline.ipynb`

Función:

- experimento comparativo de bandit offline sobre un flujo cronológico reducido

Inputs:

- `data/processed/interactions_train.parquet`
- `data/processed/interactions_valid.parquet`
- `data/processed/interactions_test.parquet`

Procesamiento documentado:

- construcción del stream temporal
- definición de recompensa binaria
- simulación de políticas

Artefactos esperados:

- `reports/tables/bandit_metrics.csv`

### `05c_model_hybrid.ipynb`

Funcion:

- combinacion de score colaborativo y score de contenido

Inputs:

- `data/processed/interactions_train.parquet`
- `data/processed/interactions_test.parquet`
- `data/processed/item_features.parquet`

Procesamiento documentado:

- construccion de subconjunto comparable con CF
- entrenamiento de item-kNN
- similitud coseno sobre features de contenido
- score hibrido final
- evaluacion sobre test

Artefactos esperados:

- `models/hybrid_recommendations.parquet`
- `reports/tables/hybrid_metrics.csv`

### `06_evaluation_and_error_analysis.ipynb`

Funcion:

- consolidacion de metricas y revision final de resultados guardados por notebooks previos

Inputs:

- metricas generadas por notebooks `05a`, `05b` y `05c`
- recomendaciones guardadas en `models/`
- interacciones procesadas

Procesamiento documentado:

- consolidacion de metricas
- segmentacion de usuarios e items
- inspeccion cualitativa de recomendaciones
- persistencia de conclusiones resumidas

Artefactos esperados:

- `reports/tables/model_comparison.csv`
- `reports/tables/final_conclusions.csv`

## 9. Orden de ejecución documentado

El orden actual de uso de notebooks es el siguiente:

1. `01_eda_ratings_and_catalog.ipynb`
2. `02_eda_content_enrichment_imdb.ipynb`
3. `03_preprocessing_interactions.ipynb`
4. `04_preprocessing_content_features.ipynb`
5. `05a_model_cf_item_user.ipynb`
6. `05b_model_bandit_offline.ipynb`
7. `05c_model_hybrid.ipynb`
8. `06_evaluation_and_error_analysis.ipynb`

## 10. Relación con otros documentos

Este documento se limita a describir estructura, inputs, outputs y notebooks implementados.

Las conclusiones del análisis, la lectura de resultados y la interpretación de métricas quedarán documentadas aparte, usando los outputs ya guardados en:

- `reports/tables/`
- `reports/figures/`
- `models/`
- `data/processed/`
