# Plan Técnico de Notebooks para el Sistema de Recomendación

## 1. Objetivo

Este documento organiza el trabajo del proyecto en notebooks reproducibles y ordenados para cubrir:

1. Configuración del proyecto y análisis de datos.
2. Preparación de datos para deep learning.
3. Implementación de un modelo de recomendación.
4. Evaluación y análisis de resultados.
5. Documentación y presentación del sistema.

En esta versión del plan, **no es necesario crear notebooks dedicados para setup ni para presentación final**. En lugar de eso:

- la configuración mínima del entorno se cubre con `requirements.txt`
- la documentación final se cubre con `README.md`, `docs/` y los artefactos generados en `reports/`

La propuesta está diseñada para la data ya disponible en el repositorio:

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

## 2. Lectura técnica de la data disponible

### MovieLens

La carpeta `data/mov_lens` corresponde al dataset `ml-32m`:

- `32,000,204` ratings
- `2,000,072` tags
- `87,585` películas
- `200,948` usuarios

Tablas principales:

- `ratings.csv`: interacciones usuario-item con `userId`, `movieId`, `rating`, `timestamp`
- `movies.csv`: catálogo base con `title` y `genres`
- `tags.csv`: señales textuales débiles aportadas por usuarios
- `links.csv`: puente hacia `IMDb` y `TMDb`

### IMDb

La carpeta `data/imdb` aporta metadatos ricos de contenido:

- `title.basics`: tipo de título, año, duración, géneros
- `title.ratings`: rating global y número de votos
- `title.crew`: directores y guionistas
- `title.principals`: cast principal y roles
- `name.basics`: nombres y profesiones de personas
- `title.akas`: títulos alternativos e idioma/región
- `title.episode`: útil para filtrar series/episodios si se desea trabajar sólo con películas

### Valor del cruce MovieLens + IMDb

Este proyecto tiene una ventaja clara: `MovieLens` resuelve el problema de interacciones y `IMDb` aporta features de contenido. Por eso, la opción más sólida para la entrega es un **sistema híbrido**, dejando CF puro y bandits como líneas comparativas.

Join recomendado:

- usar `links.csv.imdbId`
- construir `tconst = 'tt' + imdbId` con padding a la izquierda si es necesario
- unir contra las tablas `title.*` de IMDb

## 3. Estructura profesional recomendada del repositorio

```text
dmc-tp1-rec-sys/
├── data/
│   ├── imdb/
│   └── mov_lens/
├── docs/
│   └── notebooks_plan.md
├── notebooks/
│   ├── 01_eda_ratings_and_catalog.ipynb
│   ├── 02_eda_content_enrichment_imdb.ipynb
│   ├── 03_preprocessing_interactions.ipynb
│   ├── 04_preprocessing_content_features.ipynb
│   ├── 05a_model_cf_item_user.ipynb
│   ├── 05b_model_bandit_offline.ipynb
│   ├── 05c_model_hybrid.ipynb
│   ├── 06_prepare_data_for_deep_learning.ipynb
│   └── 07_evaluation_and_error_analysis.ipynb
├── reports/
│   ├── figures/
│   ├── tables/
│   └── presentation/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── utils/
├── test/
│   └── preprocessing.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

Nota:

- `test/preprocessing.ipynb` puede servir como base para dividir el preprocesamiento entre los notebooks `03` y `04`.
- `requirements.txt` reemplaza la necesidad de un notebook exclusivo de configuración.

## 4. Secuencia recomendada de notebooks

### Entorno mínimo recomendado

Antes de abrir notebooks, basta con preparar un entorno simple:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Paquetes mínimos incluidos en `requirements.txt`:

- manipulación de datos: `pandas`, `numpy`, `pyarrow`
- álgebra y sparse matrices: `scipy`
- modelado y features: `scikit-learn`
- visualización: `matplotlib`, `seaborn`
- notebooks: `jupyterlab`, `ipykernel`
- utilidades: `tqdm`

Nota:

- si más adelante entrenan un modelo neural completo, pueden agregar `torch` como dependencia opcional

### Notebook `01_eda_ratings_and_catalog.ipynb`

### Objetivo

Realizar análisis exploratorio de las interacciones y del catálogo base.

### Entradas

- `ratings.csv`
- `movies.csv`
- `tags.csv`
- `links.csv`

### Preguntas que debe responder

- ¿Cómo se distribuyen los ratings?
- ¿Cuántos ratings por usuario e ítem hay?
- ¿Qué tan dispersa es la matriz usuario-item?
- ¿Qué géneros dominan el catálogo?
- ¿Qué sesgos existen hacia ítems populares?
- ¿Qué cobertura tienen los tags?

### Análisis mínimos

- Histograma de ratings
- Ratings por usuario
- Ratings por película
- Long tail de popularidad
- Géneros más frecuentes
- Porcentaje de películas con tags
- Evolución temporal de ratings

### Salidas

- Conclusiones de sparsity, cold-start y popularidad

### Artefactos sugeridos

- `reports/figures/eda_rating_distribution.png`
- `reports/figures/eda_user_activity.png`
- `reports/figures/eda_item_popularity.png`
- `reports/figures/eda_genres.png`

### Notebook `02_eda_content_enrichment_imdb.ipynb`

### Objetivo

Analizar la calidad y utilidad de los metadatos de IMDb para enriquecer el recomendador.

### Entradas

- `links.csv`
- `title.basics.tsv.gz`
- `title.ratings.tsv.gz`
- `title.crew.tsv.gz`
- `title.principals.tsv.gz`
- `name.basics.tsv.gz`

### Preguntas que debe responder

- ¿Qué porcentaje del catálogo de MovieLens puede mapearse a IMDb?
- ¿Qué variables de IMDb tienen mejor cobertura?
- ¿Cuántos títulos son realmente `movie` versus otros `titleType`?
- ¿Qué features conviene usar para el modelo híbrido?

### Análisis mínimos

- Cobertura del join MovieLens-IMDb
- Distribución de `titleType`
- Distribución de `runtimeMinutes`
- Distribución de `startYear`
- Géneros IMDb versus géneros MovieLens
- Top directores, escritores y cast más frecuentes
- Cobertura de `numVotes` y `averageRating`

### Salidas

- Selección explícita de features de contenido
- Justificación de filtrado a películas si aplica

### Artefactos sugeridos

- `reports/tables/imdb_join_coverage.csv`
- `reports/figures/eda_imdb_type_distribution.png`
- `reports/figures/eda_runtime_distribution.png`

### Notebook `03_preprocessing_interactions.ipynb`

### Objetivo

Preparar la matriz de interacciones para modelos de recomendación.

### Entradas

- `ratings.csv`
- `movies.csv`

### Tareas

- Limpiar duplicados si existieran.
- Convertir timestamps a fechas.
- Definir filtros mínimos:
  - usuarios con al menos `N` ratings
  - ítems con al menos `M` ratings
- Elegir split temporal recomendado:
  - train: histórico
  - validation: tramo intermedio
  - test: tramo más reciente
- Construir matriz dispersa usuario-item.
- Generar baseline de popularidad.

### Salidas

- `train/validation/test`
- Mapeos `userId -> user_idx`
- Mapeos `movieId -> item_idx`

### Artefactos sugeridos

- `data/processed/interactions_train.parquet`
- `data/processed/interactions_valid.parquet`
- `data/processed/interactions_test.parquet`
- `data/processed/user_index.parquet`
- `data/processed/item_index.parquet`

### Notebook `04_preprocessing_content_features.ipynb`

### Objetivo

Construir variables de contenido útiles para modelos híbridos o deep learning.

### Entradas

- `movies.csv`
- `tags.csv`
- `links.csv`
- tablas de IMDb integradas

### Features recomendadas

- géneros MovieLens en multi-hot encoding
- géneros IMDb en multi-hot encoding
- `titleType`
- `startYear`
- `runtimeMinutes`
- `averageRating` de IMDb
- `numVotes`
- directores más frecuentes
- escritores más frecuentes
- actores/principals más frecuentes
- tags procesados con TF-IDF o vocabulario controlado

### Tareas

- Normalizar campos faltantes de IMDb (`\N`).
- Reducir cardinalidad de personas:
  - top-K directores
  - top-K actores
  - top-K escritores
- Codificar variables categóricas.
- Escalar variables numéricas.
- Generar matriz final de features por ítem.

### Salidas

- Tabla maestra de ítems enriquecidos
- Matriz de features lista para entrenamiento

### Artefactos sugeridos

- `data/processed/items_master.parquet`
- `data/processed/item_features.parquet`
- `data/processed/tag_features.parquet`

### Notebook `05a_model_cf_item_user.ipynb`

### Objetivo

Implementar un baseline fuerte de filtrado colaborativo.

### Recomendación

Priorizar **item-based CF** sobre user-based CF por escalabilidad y estabilidad con una matriz grande y dispersa.

### Alternativas válidas

- similitud coseno item-item
- similitud usuario-usuario
- kNN sobre matriz dispersa
- factorización matricial como baseline adicional

### Tareas

- Construir vecinos similares.
- Generar recomendaciones top-N.
- Comparar contra popularidad.

### Métricas

- Precision@K
- Recall@K
- MAP@K
- NDCG@K
- Coverage

### Artefactos sugeridos

- `models/cf_item_similarity.parquet`
- `reports/tables/cf_metrics.csv`

### Notebook `05b_model_bandit_offline.ipynb`

### Objetivo

Simular una estrategia de recomendación secuencial tipo bandit usando feedback histórico.

### Observación importante

Con esta data el bandit no es el camino más natural para la entrega principal, porque no existe un entorno online real. Si se usa, conviene presentarlo como **experimento comparativo offline**.

### Enfoque sugerido

- convertir ratings a recompensa binaria o escalada
- ordenar eventos por tiempo
- usar contexto del ítem y/o del usuario
- comparar políticas:
  - epsilon-greedy
  - UCB
  - Thompson Sampling

### Métricas

- cumulative reward
- average reward
- regret estimado

### Artefactos sugeridos

- `reports/tables/bandit_metrics.csv`
- `reports/figures/bandit_reward_curve.png`

### Notebook `05c_model_hybrid.ipynb`

### Objetivo

Construir el modelo principal recomendado del proyecto.

### Recomendación central

Usar un **sistema híbrido** que combine:

- señal colaborativa de `ratings.csv`
- features de contenido provenientes de `movies.csv`, `tags.csv` e `IMDb`

### Opciones de implementación

- score híbrido lineal:
  - `score_final = alpha * score_cf + (1 - alpha) * score_content`
- reranking:
  - CF genera candidatos
  - contenido reranquea
- modelo supervisado sobre features usuario-item

### Ventajas para esta entrega

- aprovecha los metadatos ricos ya disponibles
- mitiga el cold-start de ítems
- conecta naturalmente con la parte de deep learning

### Métricas

- Precision@K
- Recall@K
- NDCG@K
- Coverage
- Novelty
- Diversity

### Artefactos sugeridos

- `models/hybrid_candidates.parquet`
- `models/hybrid_scores.parquet`
- `reports/tables/hybrid_metrics.csv`

### Notebook `06_prepare_data_for_deep_learning.ipynb`

### Objetivo

Dejar los datos listos para una extensión neural del sistema.

### Casos de uso

- Neural Collaborative Filtering
- Two-Tower Retrieval
- MLP con features de usuario e ítem
- secuencias temporales de consumo

### Tareas

- crear índices enteros compactos de usuarios e ítems
- preparar embeddings IDs
- generar ejemplos negativos para training
- separar features densas y categóricas
- construir tensores o tablas finales para PyTorch/TensorFlow

### Salidas

- datasets listos para entrenamiento deep learning

### Artefactos sugeridos

- `data/processed/dl_train.parquet`
- `data/processed/dl_valid.parquet`
- `data/processed/dl_test.parquet`
- `data/processed/embedding_maps.json`

### Notebook `07_evaluation_and_error_analysis.ipynb`

### Objetivo

Consolidar la evaluación final y comparar modelos.

### Modelos a comparar

- popularidad
- CF item-based o user-based
- bandit offline
- híbrido
- opcional: modelo deep learning

### Evaluaciones mínimas

- ranking metrics por modelo
- análisis por segmento de usuarios
- análisis por popularidad del ítem
- desempeño en cold-start relativo
- comparación cualitativa de recomendaciones

### Análisis de error recomendado

- usuarios muy activos versus poco activos
- películas populares versus cola larga
- errores por género
- impacto de agregar features de IMDb

### Artefactos sugeridos

- `reports/tables/model_comparison.csv`
- `reports/figures/metrics_comparison.png`
- `reports/figures/cold_start_analysis.png`

## 5. Camino recomendado para cumplir rápido y bien

Si el objetivo es maximizar calidad técnica con el menor riesgo, el orden recomendado es:

1. `01_eda_ratings_and_catalog.ipynb`
2. `02_eda_content_enrichment_imdb.ipynb`
3. `03_preprocessing_interactions.ipynb`
4. `04_preprocessing_content_features.ipynb`
5. `05a_model_cf_item_user.ipynb`
6. `05c_model_hybrid.ipynb`
7. `06_prepare_data_for_deep_learning.ipynb`
8. `07_evaluation_and_error_analysis.ipynb`

El notebook `05b_model_bandit_offline.ipynb` se recomienda sólo si el curso exige explorar bandits o si quieren presentar una comparación adicional.

## 6. Recomendación metodológica final

Para esta data, la propuesta más defendible es:

- baseline: popularidad
- baseline fuerte: CF item-based
- modelo principal: híbrido CF + contenido IMDb/MovieLens
- extensión: preparación para deep learning y, si alcanza el tiempo, un modelo neural simple

Razones:

- la matriz de ratings es muy grande y útil para CF
- IMDb aporta features ricas para contenido
- el híbrido permite una mejor historia técnica y mejores resultados esperados
- deep learning queda bien justificado como extensión natural y no como obligación artificial

## 7. Checklist de cumplimiento por entregable

- `Configuración del proyecto y análisis de datos`
  - `requirements.txt`
  - notebook `01`, `02`
- `Preparación de datos para deep learning`
  - notebooks `03`, `04`, `06`
- `Implementar modelo de recomendación`
  - notebook `05a`, `05b` o `05c`
- `Evaluación y análisis de resultados`
  - notebook `07`
- `Documentación y presentación del sistema`
  - `README.md`
  - `docs/`
  - `reports/figures` y `reports/presentation`

## 8. Siguiente paso recomendado

Tomar `test/preprocessing.ipynb` como punto de partida y separarlo en:

- `03_preprocessing_interactions.ipynb`
- `04_preprocessing_content_features.ipynb`

Después construir primero `05a_model_cf_item_user.ipynb` y luego `05c_model_hybrid.ipynb`.
