# JustIA – Caso Práctico Unidad 2 (IA Aplicada al Desarrollo de Software)
Stanly Leon Calle Samper · Uniasturias · Especialización en Desarrollo de Software

Prototipo JustIA (Consultorio Virtual) con modelos de lenguaje preentrenados: clasificación de consultas, extracción de entidades jurídicas, preguntas y respuestas con fuente y agrupación de textos.

| Actividad | Qué hace | Modelo |
|---|---|---|
| 1. Clasificación con BETO | Fine-tuning para clasificar fragmentos jurídicos en 8 áreas del derecho | `dccuchile/bert-base-spanish-wwm-cased` |
| 2. NER jurídico | spaCy + EntityRuler para normas, tipos de violencia, delitos, fechas y autoridades; visualización con displacy | `es_core_news_md` |
| 3. Preguntas y respuestas | Pipeline en Haystack (DPR + Reader): responde preguntas sobre la base de conocimiento citando la fuente | `paraphrase-multilingual-mpnet-base-v2` + BETO SQuAD-es |
| 4. Embeddings y clusters | Sentence-BERT + KMeans + visualización con t-SNE | `paraphrase-multilingual-MiniLM-L12-v2` |

## Datos (`datos/`)
- `fragmentos_juridicos.csv`: 246 fragmentos clasificados en 8 áreas (actividades 1 y 4).
- `textos_ner.json`: 10 sentencias y resoluciones breves (actividad 2).
- `base_conocimiento/`: 8 resúmenes de normas colombianas con su fuente (actividad 3).

Todos los fragmentos, nombres y casos son **simulados con fines académicos**.

## Cómo ejecutar
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download es_core_news_md

python actividad1_beto/act1_finetune_beto.py      # entrena y evalúa (~3 min)
python actividad2_ner/act2_ner_spacy.py           # luego abrir http://localhost:5001
python actividad3_qa/act3_qa_rag.py               # escribir preguntas, 'salir' para terminar
python actividad4_embeddings/act4_clusters.py     # guarda el gráfico en resultados/actividad4/
```
