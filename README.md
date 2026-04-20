# dmc-tp1-rec-sys

Sistema de recomendación construido sobre `MovieLens 32M` enriquecido con metadatos de `IMDb`.

## Plan técnico

La guía de trabajo por notebooks está en [docs/notebooks_plan.md](/Users/chperezpelaez/Documents/Github/dmc-tp1-rec-sys/docs/notebooks_plan.md).

## Entorno mínimo

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets disponibles

- `data/mov_lens`: ratings, movies, tags y links de MovieLens
- `data/imdb`: metadatos ricos de títulos, ratings, crew, principals y nombres

## Ruta sugerida

1. EDA de interacciones y catálogo
2. Enriquecimiento con IMDb
3. Preprocesamiento
4. Modelo base CF
5. Modelo híbrido
6. Evaluación y cierre del reporte
