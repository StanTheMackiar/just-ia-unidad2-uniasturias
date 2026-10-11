"""
JustIA - Unidad 2 - Actividad 3
Stanly Leon Calle Samper

Prototipo de preguntas y respuestas juridicas con fuente, hecho con una
pipeline de Haystack (DPR + Reader).

La base de conocimiento son 8 resumenes de leyes colombianas en
datos/base_conocimiento/; cada parrafo se guarda como un Document con su
titulo, norma y URL. El retriever denso (paraphrase-multilingual-mpnet-base-v2)
busca los 3 pasajes mas parecidos a la pregunta y el reader, una version de
BETO ajustada para preguntas y respuestas en espanol (SQuAD-es), extrae el
fragmento que responde. El reader es un componente propio de Haystack porque
el ExtractiveReader de Haystack no le pasa token_type_ids al modelo y BERT los
necesita. Si ningun pasaje se parece lo suficiente, no responde y remite a un
practicante.

Ejecutar: python actividad3_qa/act3_qa_rag.py
"""

from pathlib import Path

import torch
from haystack import Document, Pipeline, component
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.dataclasses import ExtractedAnswer
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder)
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

MODELO_RETRIEVER = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MODELO_READER = "mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
UMBRAL = 0.28
RAIZ = Path(__file__).resolve().parent.parent


@component
class LectorBETO:
    def __init__(self, modelo):
        self.tokenizer = AutoTokenizer.from_pretrained(modelo)
        self.modelo = AutoModelForQuestionAnswering.from_pretrained(modelo)

    def leer(self, pregunta, contexto):
        entrada = self.tokenizer(pregunta, contexto, return_tensors="pt", truncation=True,
                                 max_length=384, return_offsets_mapping=True)
        posiciones = entrada.pop("offset_mapping")[0]
        with torch.no_grad():
            salida = self.modelo(**entrada)
        p_inicio = torch.softmax(salida.start_logits, dim=1)[0]
        p_fin = torch.softmax(salida.end_logits, dim=1)[0]
        inicio, fin = int(p_inicio.argmax()), int(p_fin.argmax())
        partes = entrada.sequence_ids(0)
        if fin < inicio or partes[inicio] != 1 or partes[fin] != 1:
            return "", 0.0
        return contexto[posiciones[inicio][0]:posiciones[fin][1]], float(p_inicio[inicio] * p_fin[fin])

    @component.output_types(answers=list[ExtractedAnswer])
    def run(self, query: str, documents: list[Document]):
        respuestas = []
        for doc in documents:
            texto, confianza = self.leer(query, doc.content)
            if texto:
                respuestas.append(ExtractedAnswer(query=query, score=confianza, data=texto, document=doc))
        respuestas.sort(key=lambda r: r.score, reverse=True)
        return {"answers": respuestas}


documentos = []
for archivo in sorted((RAIZ / "datos/base_conocimiento").glob("*.txt")):
    meta, cuerpo = {}, []
    for linea in archivo.read_text(encoding="utf-8").splitlines():
        clave = linea.split(":")[0]
        if clave in ("TITULO", "FUENTE", "URL", "NOTA"):
            meta[clave.lower()] = linea[len(clave) + 1:].strip()
        else:
            cuerpo.append(linea)
    for parrafo in "\n".join(cuerpo).split("\n\n"):
        if parrafo.strip():
            documentos.append(Document(content=parrafo.strip(), meta=meta))

store = InMemoryDocumentStore(embedding_similarity_function="cosine")
indexador = SentenceTransformersDocumentEmbedder(model=MODELO_RETRIEVER)
store.write_documents(indexador.run(documentos)["documents"])
print(f"Base de conocimiento: {store.count_documents()} pasajes")

pipeline = Pipeline()
pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model=MODELO_RETRIEVER, progress_bar=False))
pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=store, top_k=3))
pipeline.add_component("reader", LectorBETO(MODELO_READER))
pipeline.connect("embedder.embedding", "retriever.query_embedding")
pipeline.connect("retriever.documents", "reader.documents")


def responder(pregunta):
    resultado = pipeline.run({"embedder": {"text": pregunta}, "reader": {"query": pregunta}},
                             include_outputs_from={"retriever"})
    pasajes = resultado["retriever"]["documents"]
    respuestas = resultado["reader"]["answers"]

    print("-" * 70)
    if pasajes[0].score < UMBRAL:
        print("No encontré información sobre esto. Su consulta se remite a un practicante.")
        return

    if respuestas:
        mejor = respuestas[0]
        pasaje = mejor.document
        print(f"Respuesta: {mejor.data}")
        print(f"Confianza: {mejor.score:.0%} | Similitud del pasaje: {pasaje.score:.2f}")
        print(f"Contexto: {pasaje.content}")
    else:
        pasaje = pasajes[0]
        print("Respuesta: no pude extraer una respuesta puntual, este es el pasaje más relacionado:")
        print(pasaje.content)
    print(f"Fuente: {pasaje.meta['titulo']}")
    print(f"        {pasaje.meta['fuente']}")
    print(f"        {pasaje.meta['url']}")
    print("Aviso: orientación preliminar, debe ser validada por un practicante del consultorio.")


print("\nJustIA - Preguntas y respuestas (escriba 'salir' para terminar)")
while True:
    pregunta = input("\nSu pregunta: ").strip()
    if pregunta.lower() == "salir":
        break
    if pregunta:
        responder(pregunta)
