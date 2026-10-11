"""
JustIA - Unidad 2 - Actividad 1
Stanly Leon Calle Samper

Clasifica fragmentos juridicos por area del derecho (familia, laboral, penal, etc.)
haciendo fine-tuning de BETO (dccuchile/bert-base-spanish-wwm-cased).

Se limpian los textos, se dividen en entrenamiento (80%) y prueba (20%), se
tokenizan con el tokenizer de BETO y se entrena con Trainer de Hugging Face.
Al final se muestran accuracy, F1 y algunas predicciones con frases nuevas.

Ejecutar: python actividad1_beto/act1_finetune_beto.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

MODELO = "dccuchile/bert-base-spanish-wwm-cased"
SEMILLA = 42
RAIZ = Path(__file__).resolve().parent.parent
torch.manual_seed(SEMILLA)


def limpiar(texto):
    texto = re.sub(r"\((simulad[oa]|adaptado|texto adaptado)\)", " ", texto, flags=re.I)
    texto = re.sub(r"https?://\S+", " ", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


df = pd.read_csv(RAIZ / "datos/fragmentos_juridicos.csv")
df["texto"] = df["texto"].apply(limpiar)

areas = sorted(df["categoria"].unique())
area_a_id = {a: i for i, a in enumerate(areas)}
id_a_area = {i: a for a, i in area_a_id.items()}
df["label"] = df["categoria"].map(area_a_id)

print("Total de fragmentos:", len(df))
print(df["categoria"].value_counts(), "\n")

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEMILLA)
print(f"Entrenamiento: {len(train_df)} | Prueba: {len(test_df)}\n")

tokenizer = AutoTokenizer.from_pretrained(MODELO)

ejemplo = train_df["texto"].iloc[0]
print("Ejemplo de texto:", ejemplo)
print("Tokens:", tokenizer.tokenize(ejemplo), "\n")


class FragmentosDataset(torch.utils.data.Dataset):
    def __init__(self, textos, labels):
        self.enc = tokenizer(list(textos), truncation=True, padding="max_length", max_length=128)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


train_ds = FragmentosDataset(train_df["texto"], train_df["label"])
test_ds = FragmentosDataset(test_df["texto"], test_df["label"])

modelo = AutoModelForSequenceClassification.from_pretrained(
    MODELO, num_labels=len(areas), id2label=id_a_area, label2id=area_a_id)


def calcular_metricas(pred):
    y_pred = np.argmax(pred.predictions, axis=1)
    return {"accuracy": accuracy_score(pred.label_ids, y_pred),
            "f1": f1_score(pred.label_ids, y_pred, average="macro")}


args = TrainingArguments(
    output_dir=str(RAIZ / "modelos/checkpoints"),
    num_train_epochs=5,
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=10,
    seed=SEMILLA,
    report_to="none",
)

trainer = Trainer(model=modelo, args=args, train_dataset=train_ds,
                  eval_dataset=test_ds, compute_metrics=calcular_metricas)
trainer.train()

pred = trainer.predict(test_ds)
y_pred = np.argmax(pred.predictions, axis=1)
y_real = test_df["label"].values

print("\n===== Resultados en el conjunto de prueba =====")
print(f"Accuracy: {accuracy_score(y_real, y_pred):.4f}")
print(f"F1 macro: {f1_score(y_real, y_pred, average='macro'):.4f}\n")
print(classification_report(y_real, y_pred, target_names=areas, zero_division=0))

ejemplos = [
    "Me despidieron estando incapacitado y no me pagaron las vacaciones",
    "Mi exmarido no me deja ver a los niños y tampoco paga la cuota",
    "La Fiscalía legalizó la captura del sospechoso del atraco",
    "Nos desplazaron de la vereda y queremos volver a nuestra tierra",
    "Él controla mi plata y me quitó la tarjeta del banco",
]

print("===== Ejemplos de prediccion =====")
modelo.eval()
for frase in ejemplos:
    entrada = tokenizer(limpiar(frase), return_tensors="pt").to(modelo.device)
    with torch.no_grad():
        probs = torch.softmax(modelo(**entrada).logits, dim=1)[0]
    i = int(probs.argmax())
    print(f"{frase}\n  -> {id_a_area[i]} ({probs[i]:.1%})")

trainer.save_model(str(RAIZ / "modelos/beto_justia"))
tokenizer.save_pretrained(RAIZ / "modelos/beto_justia")
