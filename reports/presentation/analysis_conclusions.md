# Conclusiones del Análisis del Sistema de Recomendación

## 1. Resumen ejecutivo

El análisis realizado sobre `MovieLens 32M` enriquecido con metadatos de `IMDb` muestra que el problema de recomendación tiene una matriz usuario-item extremadamente dispersa, pero al mismo tiempo dispone de una cobertura muy alta de metadatos de contenido. En consecuencia, el uso de señales colaborativas por sí solo resulta útil como baseline, mientras que la incorporación de features de contenido mejora el desempeño del sistema y amplía la capacidad de generalización sobre el catálogo.

Los resultados obtenidos indican que el modelo híbrido supera al baseline colaborativo en las métricas de ranking evaluadas y constituye el mejor resultado observado dentro del conjunto de notebooks implementados.

## 2. Hallazgos del análisis exploratorio

### Volumen de datos

Según `reports/tables/eda_catalog_summary.csv`:

- ratings totales: `32,000,204`
- usuarios: `200,948`
- películas con ratings: `84,432`
- películas en el catálogo: `87,585`
- tags totales: `2,000,072`
- películas con tags: `51,323`

### Estructura de la matriz de interacción

Según `reports/tables/eda_interaction_metrics.csv`:

- sparsity de la matriz usuario-item: `0.998114`
- mediana de ratings por usuario: `73`
- mediana de ratings por ítem: `5`

Interpretación:

- la matriz presenta una dispersión muy alta
- existe una fuerte asimetría entre usuarios/ítems activos y la cola larga del catálogo
- las recomendaciones basadas sólo en co-ocurrencia enfrentan un contexto desafiante

### Cobertura de tags y catálogo

Según `reports/tables/eda_catalog_coverage.csv`:

- proporción de películas con tags: `0.58598`
- géneros distintos en MovieLens: `19`
- películas con vínculo a IMDb: `1.0`

Interpretación:

- más de la mitad del catálogo dispone de tags
- el enlace hacia IMDb es completo a nivel de `links.csv`
- la información textual y de metadatos resulta suficiente para enriquecer el sistema

## 3. Hallazgos del enriquecimiento con IMDb

### Cobertura del join MovieLens-IMDb

Según `reports/tables/imdb_join_coverage.csv`:

- películas del catálogo: `87,585`
- películas con `tconst`: `87,585` (`100%`)
- películas con título IMDb: `87,359` (`99.74%`)
- películas con rating IMDb: `87,223` (`99.59%`)
- películas con duración: `86,894` (`99.21%`)

### Calidad de features de contenido

Según `reports/tables/imdb_feature_quality.csv` y `reports/tables/item_feature_quality.csv`:

- cobertura de `titleType`: `99.74%`
- cobertura de `runtimeMinutes`: `99.21%`
- cobertura de `averageRating`: `99.59%`
- cobertura de `numVotes`: `99.59%`
- cobertura de `directors_list`: `99.30%`
- cobertura de `principal_names`: `99.65%`
- cobertura de `writers_list`: `93.13%`

Interpretación:

- la calidad del enriquecimiento es alta
- los metadatos de IMDb no sólo completan el catálogo, sino que ofrecen señales estructuradas muy consistentes
- el sistema dispone de una base sólida para construir representaciones de contenido

## 4. Hallazgos del preprocesamiento de interacciones

Según `reports/tables/interaction_preprocessing_summary.csv`:

- interacciones originales: `32,000,204`
- usuarios originales: `200,948`
- ítems originales con rating: `84,432`
- interacciones tras filtrado: `31,722,400`
- usuarios tras filtrado: `200,763`
- ítems tras filtrado: `23,339`

Según `reports/tables/interaction_split_summary.csv`:

- train: `25,377,920` filas
- validation: `457,592` filas
- test: `301,839` filas

Interpretación:

- el filtrado conserva casi toda la masa de interacciones
- el mayor recorte se observa del lado de los ítems, lo que confirma la existencia de una cola larga muy marcada
- el conjunto final sigue siendo lo suficientemente grande para entrenar y comparar modelos

## 5. Resultados de modelado

### Baseline colaborativo

Según `reports/tables/cf_metrics.csv`:

- `Precision@10`: `0.0446`
- `Recall@10`: `0.0260`
- `NDCG@10`: `0.0539`
- usuarios evaluados: `881`

Lectura:

- el baseline colaborativo logra capturar parte de la señal en ítems populares y en zonas densas del espacio de interacción
- su cobertura de usuarios evaluados es menor que la del híbrido dentro del prototipo ejecutado

### Modelo híbrido

Según `reports/tables/hybrid_metrics.csv`:

- `Precision@10`: `0.0589`
- `Recall@10`: `0.0331`
- `NDCG@10`: `0.0687`
- usuarios evaluados: `3,864`

Comparación directa contra CF:

- mejora absoluta en `Precision@10`: `+0.0143`
- mejora absoluta en `Recall@10`: `+0.0071`
- mejora absoluta en `NDCG@10`: `+0.0149`

Lectura:

- el modelo híbrido obtiene mejores métricas en las tres medidas de ranking consideradas
- también logra evaluar una mayor cantidad de usuarios dentro del esquema aplicado
- el enriquecimiento de contenido aporta valor observable sobre el baseline colaborativo

### Experimento bandit offline

Según `reports/tables/bandit_metrics.csv`:

- política `greedy`
  - matched events: `89`
  - average reward: `0.6180`
  - estimated regret: `0.2544`
- política `epsilon_greedy`
  - matched events: `122`
  - average reward: `0.6393`
  - estimated regret: `0.2330`

Lectura:

- el experimento bandit se mantiene como comparación offline
- `epsilon_greedy` supera a `greedy` en reward medio y reduce regret estimado
- su interpretación debe mantenerse separada de las métricas de ranking tradicionales

## 6. Conclusiones principales

1. El problema presenta una matriz de interacción extremadamente dispersa, con fuerte cola larga en los ítems.
2. La conexión entre MovieLens e IMDb ofrece una cobertura casi completa y metadatos muy consistentes.
3. El enriquecimiento de contenido queda bien sustentado por la disponibilidad de géneros, ratings, duración, directores, guionistas y cast principal.
4. El baseline colaborativo item-based funciona como referencia válida, pero no es el mejor resultado observado.
5. El modelo híbrido muestra la mejor performance dentro de los resultados guardados, con mejoras en `Precision@10`, `Recall@10` y `NDCG@10`.
6. El experimento bandit aporta una comparación complementaria, pero no reemplaza la evaluación principal del sistema recomendador.

## 7. Evidencia disponible

Tablas principales:

- [eda_catalog_summary.csv](/Users/chperezpelaez/Documents/Github/dmc-tp1-rec-sys/reports/tables/eda_catalog_summary.csv)
- [eda_interaction_metrics.csv](/Users/chperezpelaez/Documents/Github/dmc-tp1-rec-sys/reports/tables/eda_interaction_metrics.csv)
- [imdb_join_coverage.csv](/Users/chperezpelaez/Documents/Github/dmc-tp1-rec-sys/reports/tables/imdb_join_coverage.csv)
- [item_feature_quality.csv](/Users/chperezpelaez/Documents/Github/dmc-tp1-rec-sys/reports/tables/item_feature_quality.csv)
- [cf_metrics.csv](/Users/chperezpelaez/Documents/Github/dmc-tp1-rec-sys/reports/tables/cf_metrics.csv)
- [hybrid_metrics.csv](/Users/chperezpelaez/Documents/Github/dmc-tp1-rec-sys/reports/tables/hybrid_metrics.csv)
- [bandit_metrics.csv](/Users/chperezpelaez/Documents/Github/dmc-tp1-rec-sys/reports/tables/bandit_metrics.csv)
- [model_comparison.csv](/Users/chperezpelaez/Documents/Github/dmc-tp1-rec-sys/reports/tables/model_comparison.csv)

Figuras principales:

- [eda_ratings_users_items.png](/Users/chperezpelaez/Documents/Github/dmc-tp1-rec-sys/reports/figures/eda_ratings_users_items.png)
- [eda_genres_tags_timeline.png](/Users/chperezpelaez/Documents/Github/dmc-tp1-rec-sys/reports/figures/eda_genres_tags_timeline.png)
- [eda_imdb_type_runtime_year.png](/Users/chperezpelaez/Documents/Github/dmc-tp1-rec-sys/reports/figures/eda_imdb_type_runtime_year.png)
- [eda_imdb_people_and_genres.png](/Users/chperezpelaez/Documents/Github/dmc-tp1-rec-sys/reports/figures/eda_imdb_people_and_genres.png)
