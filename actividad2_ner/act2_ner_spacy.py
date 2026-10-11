"""
JustIA - Unidad 2 - Actividad 2
Stanly Leon Calle Samper

Extractor de entidades (NER) para textos juridicos con spaCy.

El modelo es_core_news_md reconoce personas y lugares, pero no sabe que
"Ley 1257 de 2008" es una norma o que "violencia economica" es un tipo de
violencia. Por eso se agrega un EntityRuler con patrones para NORMA, VIOLENCIA,
DELITO, FECHA y JURISDICCION, y se aplica a 10 sentencias simuladas.
Los resultados se ven con displacy en el navegador.

Ejecutar: python actividad2_ner/act2_ner_spacy.py
Luego abrir http://localhost:5001 (Ctrl+C para terminar)
"""

import json
from pathlib import Path

import spacy
from spacy import displacy

RAIZ = Path(__file__).resolve().parent.parent
nlp = spacy.load("es_core_news_md")

meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
codigos = ["código penal", "código civil", "código sustantivo del trabajo",
           "código general del proceso", "constitución política"]
articulo = [{"LOWER": "artículo"}, {"LIKE_NUM": True}, {"LOWER": {"IN": ["de", "del"]}}, {"LOWER": "la", "OP": "?"}]

patrones = []

ley = [{"LOWER": "ley"}, {"LOWER": "estatutaria", "OP": "?"}, {"LIKE_NUM": True}, {"LOWER": "de"}, {"SHAPE": "dddd"}]
decreto = [{"LOWER": "decreto"}, {"LIKE_NUM": True}, {"LOWER": "de"}, {"SHAPE": "dddd"}]
patrones.append({"label": "NORMA", "pattern": ley})
patrones.append({"label": "NORMA", "pattern": decreto})
patrones.append({"label": "NORMA", "pattern": articulo + ley})
for c in codigos:
    palabras = [{"LOWER": w} for w in c.split()]
    patrones.append({"label": "NORMA", "pattern": palabras})
    patrones.append({"label": "NORMA", "pattern": articulo + palabras})

tipos = ["intrafamiliar", "económica", "patrimonial", "psicológica", "física", "sexual"]
patrones.append({"label": "VIOLENCIA", "pattern": [{"LOWER": "violencia"}, {"LOWER": {"IN": tipos}}]})
patrones.append({"label": "VIOLENCIA", "pattern": [{"LOWER": "violencia"}, {"LOWER": "de"}, {"LOWER": "género"}]})
patrones.append({"label": "VIOLENCIA", "pattern": [{"LOWER": "acoso"}, {"LOWER": "laboral"}]})

delitos = ["feminicidio", "homicidio", "extorsión", "estafa", "hurto calificado", "lesiones personales",
           "inasistencia alimentaria", "concierto para delinquir", "desplazamiento forzado"]
for d in delitos:
    patrones.append({"label": "DELITO", "pattern": [{"LOWER": w} for w in d.split()]})

patrones.append({"label": "FECHA", "pattern": [{"LIKE_NUM": True}, {"LOWER": "de"},
                                               {"LOWER": {"IN": meses}}, {"LOWER": "de"}, {"SHAPE": "dddd"}]})
patrones.append({"label": "FECHA", "pattern": [{"LOWER": {"IN": meses}}, {"LOWER": "de"}, {"SHAPE": "dddd"}]})

patrones.append({"label": "JURISDICCION", "pattern": [{"LOWER": "juzgado"}, {"IS_TITLE": True, "OP": "+"},
                                                      {"LOWER": {"IN": ["de", "del"]}, "OP": "?"},
                                                      {"IS_TITLE": True, "OP": "*"}]})
patrones.append({"label": "JURISDICCION", "pattern": [{"LOWER": "tribunal"}, {"LOWER": "superior"},
                                                      {"LOWER": "de"}, {"IS_TITLE": True}]})
autoridades = ["Corte Constitucional", "Corte Suprema de Justicia", "Fiscalía General de la Nación",
               "Comisaría de Familia", "Unidad para las Víctimas", "Superintendencia de Industria y Comercio"]
for a in autoridades:
    patrones.append({"label": "JURISDICCION", "pattern": a})

ruler = nlp.add_pipe("entity_ruler", before="ner")
ruler.add_patterns(patrones)
print("Pipeline:", nlp.pipe_names)
print("Patrones agregados:", len(patrones))

with open(RAIZ / "datos/textos_ner.json", encoding="utf-8") as f:
    textos = json.load(f)

docs = []
for t in textos:
    doc = nlp(t["texto"])
    doc.user_data["title"] = t["id"]
    docs.append(doc)
    print(f"\n[{t['id']}]")
    for ent in doc.ents:
        print(f"  {ent.label_:<13} {ent.text}")

colores = {"NORMA": "#7aecec", "VIOLENCIA": "#ff9561", "DELITO": "#ff6b6b", "FECHA": "#bfe1d9",
           "JURISDICCION": "#c887fb", "PER": "#aa9cfc", "LOC": "#feca74"}
displacy.serve(docs, style="ent", options={"colors": colores}, port=5001, auto_select_port=True)
