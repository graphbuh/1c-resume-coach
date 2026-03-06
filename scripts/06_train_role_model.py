import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder


IN_TEXTS = os.path.join("outputs", "dedup", "vacancies_dedup.jsonl")
IN_LABELS = os.path.join("outputs", "roles", "vacancy_roles.csv")

OUT_DIR = os.path.join("outputs", "models")
OUT_MODEL = os.path.join(OUT_DIR, "role_mlp_pipeline.joblib")
OUT_LABEL_ENCODER = os.path.join(OUT_DIR, "role_label_encoder.joblib")
OUT_METRICS = os.path.join(OUT_DIR, "role_model_metrics.json")
OUT_PREDICTIONS = os.path.join(OUT_DIR, "role_model_holdout_predictions.csv")


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(IN_TEXTS):
        raise FileNotFoundError(f"Missing file: {IN_TEXTS}")
    if not os.path.exists(IN_LABELS):
        raise FileNotFoundError(f"Missing file: {IN_LABELS}")

    texts_rows = load_jsonl(IN_TEXTS)
    roles_df = pd.read_csv(IN_LABELS, encoding="utf-8-sig")

    texts_df = pd.DataFrame(texts_rows)
    texts_df["vacancy_id"] = texts_df.index

    df = texts_df.merge(
        roles_df[["vacancy_id", "role"]],
        on="vacancy_id",
        how="inner"
    )

    df = df[["vacancy_id", "text", "role"]].dropna().copy()

    print(f"[INFO] training rows: {len(df)}")
    print(f"[INFO] classes: {sorted(df['role'].unique().tolist())}")

    X = df["text"].astype(str)
    y = df["role"].astype(str)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X,
        y_encoded,
        df["vacancy_id"],
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                token_pattern=r"(?u)\b[\w\-\/\.]+\b",
                sublinear_tf=True,
            ),
        ),
        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                random_state=42,
            ),
        ),
    ])

    print("[INFO] fitting model...")
    pipeline.fit(X_train, y_train)

    y_pred_encoded = pipeline.predict(X_test)

    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

    acc = accuracy_score(y_test_labels, y_pred_labels)
    report = classification_report(
        y_test_labels,
        y_pred_labels,
        output_dict=True,
        zero_division=0
    )

    class_labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=class_labels)

    print(f"[INFO] holdout accuracy: {acc:.4f}")

    # Сохраняем модель и энкодер
    joblib.dump(pipeline, OUT_MODEL)
    joblib.dump(label_encoder, OUT_LABEL_ENCODER)

    # Сохраняем holdout predictions
    pred_df = pd.DataFrame({
        "vacancy_id": id_test.values,
        "y_true": y_test_labels,
        "y_pred": y_pred_labels,
    })
    pred_df.to_csv(OUT_PREDICTIONS, index=False, encoding="utf-8-sig")

    metrics = {
        "rows_total": int(len(df)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "accuracy": float(acc),
        "labels": class_labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "model_type": "TF-IDF + MLPClassifier",
        "files": {
            "model": OUT_MODEL,
            "label_encoder": OUT_LABEL_ENCODER,
            "holdout_predictions": OUT_PREDICTIONS,
        },
    }

    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote: {OUT_MODEL}")
    print(f"[OK] wrote: {OUT_LABEL_ENCODER}")
    print(f"[OK] wrote: {OUT_METRICS}")
    print(f"[OK] wrote: {OUT_PREDICTIONS}")


if __name__ == "__main__":
    main()