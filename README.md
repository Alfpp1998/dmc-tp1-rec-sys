# dmc-tp1-rec-sys

Sistema de recomendación construido sobre `MovieLens 32M` enriquecido con metadatos de `IMDb`.

El repositorio contiene notebooks para análisis exploratorio, preprocesamiento, modelado y evaluación, además de utilidades compartidas para carga, limpieza, cache y generación de artefactos.

## Documentación

- Documentación de notebooks y estructura de datos: [docs/notebooks_documentation.md](/Users/chperezpelaez/Documents/Github/dmc-tp1-rec-sys/docs/notebooks_documentation.md)
- Conclusiones del análisis: [reports/presentation/analysis_conclusions.md](/Users/chperezpelaez/Documents/Github/dmc-tp1-rec-sys/reports/presentation/analysis_conclusions.md)

## Estructura actual

```text
dmc-tp1-rec-sys/
├── data/
│   ├── imdb/
│   ├── mov_lens/
│   └── processed/
├── docs/
│   └── notebooks_documentation.md
├── models/
│   ├── cf_recommendations.parquet
│   └── hybrid_recommendations.parquet
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
│   ├── presentation/
│   └── tables/
├── src/
│   └── utils/
│       └── recsys_utils.py
├── .gitignore
└── requirements.txt
```

## Entorno con `pyenv`

Ejemplo usando `pyenv` con Python `3.10.13`:

```bash
pyenv install 3.10.13 -s
pyenv virtualenv 3.10.13 dmc-env
pyenv local dmc-env
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si se quiere registrar el entorno en Jupyter:

```bash
python -m ipykernel install --user --name dmc-env --display-name "Python (dmc-env)"
```

## Datasets

Los datasets no se suben al repositorio. Deben descargarse y colocarse localmente en:

- `data/imdb/`
- `data/mov_lens/`

Fuentes oficiales:

- IMDb: <https://datasets.imdbws.com/>
- MovieLens 32M: <https://grouplens.org/datasets/movielens/32m/>

Archivos utilizados:

- `data/mov_lens/ratings.csv`
- `data/mov_lens/movies.csv`
- `data/mov_lens/tags.csv`
- `data/mov_lens/links.csv`
- `data/imdb/title.basics.tsv.gz`
- `data/imdb/title.ratings.tsv.gz`
- `data/imdb/title.crew.tsv.gz`
- `data/imdb/title.principals.tsv.gz`
- `data/imdb/name.basics.tsv.gz`
- `data/imdb/title.akas.tsv.gz`
- `data/imdb/title.episode.tsv.gz`

## Notebooks

El flujo implementado en la branch actual es:

1. [01_eda_ratings_and_catalog.ipynb](/Users/chperezpelaez/Documents/Github/dmc-tp1-rec-sys/notebooks/01_eda_ratings_and_catalog.ipynb)
2. [02_eda_content_enrichment_imdb.ipynb](/Users/chperezpelaez/Documents/Github/dmc-tp1-rec-sys/notebooks/02_eda_content_enrichment_imdb.ipynb)
3. [03_preprocessing_interactions.ipynb](/Users/chperezpelaez/Documents/Github/dmc-tp1-rec-sys/notebooks/03_preprocessing_interactions.ipynb)
4. [04_preprocessing_content_features.ipynb](/Users/chperezpelaez/Documents/Github/dmc-tp1-rec-sys/notebooks/04_preprocessing_content_features.ipynb)
5. [05a_model_cf_item_user.ipynb](/Users/chperezpelaez/Documents/Github/dmc-tp1-rec-sys/notebooks/05a_model_cf_item_user.ipynb)
6. [05b_model_bandit_offline.ipynb](/Users/chperezpelaez/Documents/Github/dmc-tp1-rec-sys/notebooks/05b_model_bandit_offline.ipynb)
7. [05c_model_hybrid.ipynb](/Users/chperezpelaez/Documents/Github/dmc-tp1-rec-sys/notebooks/05c_model_hybrid.ipynb)
8. [06_evaluation_and_error_analysis.ipynb](/Users/chperezpelaez/Documents/Github/dmc-tp1-rec-sys/notebooks/06_evaluation_and_error_analysis.ipynb)

## Artefactos generados

Durante la ejecución se generan artefactos en:

- `data/processed/`: interacciones procesadas, índices, features y caches
- `reports/tables/`: tablas resumen y métricas
- `reports/figures/`: figuras exportadas desde notebooks
- `reports/presentation/`: documentos markdown de apoyo para presentación
- `models/`: recomendaciones guardadas por los modelos

## Utilidades compartidas

La lógica común del proyecto está centralizada en [src/utils/recsys_utils.py](/Users/chperezpelaez/Documents/Github/dmc-tp1-rec-sys/src/utils/recsys_utils.py).

Entre sus responsabilidades actuales están:

- carga de tablas IMDb y MovieLens
- limpieza y tipado de columnas
- logging de progreso
- cache de tablas limpias
- cache del join enriquecido MovieLens-IMDb
- construcción de interacciones y features
- métricas de evaluación

## Nota sobre `.gitignore`

El repositorio ignora:

- datasets locales
- `data/processed/`
- `models/`
- `reports/figures/`
- `reports/tables/`
- `__pycache__/`
- `.ipynb_checkpoints/`

Se mantienen versionables los notebooks, la documentación y los markdowns de `reports/presentation/`.
