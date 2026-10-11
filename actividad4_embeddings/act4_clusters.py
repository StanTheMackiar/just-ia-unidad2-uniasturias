"""
JustIA - Unidad 2 - Actividad 4
Stanly Leon Calle Samper

Agrupa fragmentos juridicos por similitud tematica usando embeddings.

Cada fragmento se convierte en un vector con Sentence-BERT
(paraphrase-multilingual-MiniLM-L12-v2) y se agrupan con KMeans en 8 clusters,
uno por area del derecho. La categoria real no se usa para agrupar, solo para
comparar despues con una tabla cruzada. Al final se reduce a 2D con t-SNE y se
guarda el grafico en resultados/actividad4/.

Ejecutar: python actividad4_embeddings/act4_clusters.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score

SEMILLA = 42
RAIZ = Path(__file__).resolve().parent.parent
K = 8

df = pd.read_csv(RAIZ / "datos/fragmentos_juridicos.csv")
print("Fragmentos:", len(df))

modelo = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embeddings = modelo.encode(df["texto"].tolist(), normalize_embeddings=True, show_progress_bar=True)
print("Tamaño de los embeddings:", embeddings.shape)

kmeans = KMeans(n_clusters=K, n_init=10, random_state=SEMILLA)
df["cluster"] = kmeans.fit_predict(embeddings)

print(f"\nSilueta: {silhouette_score(embeddings, df['cluster']):.3f}")
print(f"ARI frente a las areas reales: {adjusted_rand_score(df['categoria'], df['cluster']):.3f}")

tabla = pd.crosstab(df["cluster"], df["categoria"])
print("\nTabla cruzada cluster vs area:")
print(tabla.to_string())

print("\nResumen de cada cluster:")
for c in range(K):
    grupo = df[df["cluster"] == c]
    area = grupo["categoria"].value_counts()
    print(f"\nCluster {c} ({len(grupo)} textos) - area dominante: {area.index[0]} ({area.iloc[0] / len(grupo):.0%})")
    print("  Ejemplo:", grupo["texto"].iloc[0][:120])

tsne = TSNE(n_components=2, perplexity=30, random_state=SEMILLA)
puntos = tsne.fit_transform(embeddings)

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
for i, area in enumerate(sorted(df["categoria"].unique())):
    m = df["categoria"] == area
    ax[0].scatter(puntos[m, 0], puntos[m, 1], s=20, color=plt.cm.tab10(i), label=area)
ax[0].set_title("t-SNE por area real")
ax[0].legend(fontsize=8)

for c in range(K):
    m = df["cluster"] == c
    ax[1].scatter(puntos[m, 0], puntos[m, 1], s=20, color=plt.cm.tab10(c), label=f"Cluster {c}")
ax[1].set_title("t-SNE por cluster (KMeans k=8)")
ax[1].legend(fontsize=8)

(RAIZ / "resultados/actividad4").mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(RAIZ / "resultados/actividad4/tsne_clusters.png", dpi=150)
print("\nGrafico guardado en resultados/actividad4/tsne_clusters.png")
plt.show()
