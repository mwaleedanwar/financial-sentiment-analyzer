#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from datasets import ClassLabel, Dataset, DatasetDict, concatenate_datasets, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilbert-base-uncased"
DATASET_ID = "zeroshot/twitter-financial-news-sentiment"
OUTPUT_DIR = Path(__file__).resolve().parent / "financial_model"
MAX_LENGTH = 128
TRAIN_TEST_SPLIT = 0.2
SEED = 42
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01

ID2LABEL = {0: "Negative", 1: "Positive", 2: "Neutral"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def detect_text_column(column_names: list[str]) -> str:
    for candidate in ("text", "sentence", "tweet", "content", "message"):
        if candidate in column_names:
            return candidate
    raise ValueError(f"Could not infer text column from: {column_names}")


def detect_label_column(column_names: list[str]) -> str:
    for candidate in ("label", "labels", "sentiment", "target"):
        if candidate in column_names:
            return candidate
    raise ValueError(f"Could not infer label column from: {column_names}")


def load_merged_splits() -> Dataset:
    raw: DatasetDict = load_dataset(DATASET_ID)
    pieces = []
    for split_name in ("train", "validation", "test"):
        if split_name in raw:
            pieces.append(raw[split_name])
    if not pieces:
        raise RuntimeError(f"No splits found in {DATASET_ID}: {list(raw.keys())}")
    merged = concatenate_datasets(pieces) if len(pieces) > 1 else pieces[0]
    return merged


def compute_metrics_factory(label_list: list[int]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "f1_macro": f1_score(labels, preds, average="macro", labels=label_list),
            "f1_weighted": f1_score(labels, preds, average="weighted", labels=label_list),
            "accuracy": accuracy_score(labels, preds),
        }

    return compute_metrics


def main() -> int:
    device = resolve_device()
    print(f"Using device: {device}")

    full_ds = load_merged_splits()
    text_col = detect_text_column(full_ds.column_names)
    label_col = detect_label_column(full_ds.column_names)
    print(f"Text column: {text_col!r}, label column: {label_col!r}")

    # train_test_split(..., stratify_by_column=...) requires ClassLabel, not Value(int).
    if not isinstance(full_ds.features[label_col], ClassLabel):
        class_names = [ID2LABEL[i] for i in range(len(ID2LABEL))]
        full_ds = full_ds.cast_column(label_col, ClassLabel(names=class_names))

    split = full_ds.train_test_split(
        test_size=TRAIN_TEST_SPLIT,
        seed=SEED,
        stratify_by_column=label_col,
    )
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"Train rows: {len(train_ds)}, Eval rows: {len(eval_ds)}")

    label_values = sorted(set(train_ds[label_col]))
    if label_values != [0, 1, 2]:
        print(
            "Warning: unexpected label set "
            f"{label_values}; model still uses num_labels=3.",
            file=sys.stderr,
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def _tok(batch):
        # Labels must be copied into the batch before remove_columns drops the label column.
        encoded = tokenizer(
            batch[text_col],
            truncation=True,
            max_length=MAX_LENGTH,
        )
        encoded["labels"] = batch[label_col]
        return encoded

    tokenized_train = train_ds.map(_tok, batched=True, remove_columns=train_ds.column_names)
    tokenized_eval = eval_ds.map(_tok, batched=True, remove_columns=eval_ds.column_names)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "trainer_checkpoints"),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        seed=SEED,
        load_best_model_at_end=False,
        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_factory(label_list=[0, 1, 2]),
    )

    print("Starting training...")
    trainer.train()

    print("Running final evaluation...")
    metrics = trainer.evaluate()
    f1_macro = metrics.get("eval_f1_macro", float("nan"))
    f1_weighted = metrics.get("eval_f1_weighted", float("nan"))
    acc = metrics.get("eval_accuracy", float("nan"))
    print(f"Eval accuracy:      {acc:.4f}")
    print(f"Eval F1 (macro):    {f1_macro:.4f}  (coursework target often > 0.85)")
    print(f"Eval F1 (weighted): {f1_weighted:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Saved model + tokenizer to: {OUTPUT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
