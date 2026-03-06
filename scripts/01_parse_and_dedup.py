import re
import os
import json
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple

import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# HH разделитель в твоих файлах: "+++++++"
DELIMITER = r"\+{5,}"  # 5+ плюсов


@dataclass
class Vacancy:
    source: str                 # "hh_txt" / "telegram_jsonl"
    source_file: str            # имя файла или "tg_vacancies.jsonl"
    idx_in_source: int          # индекс блока внутри источника
    text_raw: str               # исходный текст блока
    text_norm: str              # нормализованный текст
    text_hash: str              # hash(norm)


def normalize_text(s: str) -> str:
    """
    Нормализация для дедупликации:
    - lower
    - ё->е
    - унификация 1C -> 1с
    - сжатие пробелов
    - сжатие пустых строк
    """
    if not s:
        return ""

    s = s.replace("\u00ad", "")  # мягкий перенос
    s = s.replace("ё", "е").replace("Ё", "Е")
    s = s.lower()

    # унификация 1c/1C -> 1с
    s = re.sub(r"\b1c\b", "1с", s)

    # чуть-чуть подчистим типовые "мусорные" пробелы
    s = s.replace("\t", " ")
    s = re.sub(r"[ ]{2,}", " ", s)

    # сжимаем пустые строки
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()


def text_to_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def split_blocks(file_text: str) -> List[str]:
    blocks = re.split(DELIMITER, file_text)
    blocks = [b.strip() for b in blocks if b and b.strip()]
    return blocks


def load_txt_files(raw_dir: str) -> List[Vacancy]:
    """
    Загружает *.txt из data/raw и режет на блоки по +++++++
    """
    items: List[Vacancy] = []

    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"raw_dir not found: {raw_dir}")

    for fname in sorted(os.listdir(raw_dir)):
        if not fname.lower().endswith(".txt"):
            continue

        path = os.path.join(raw_dir, fname)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        blocks = split_blocks(content)

        for i, block in enumerate(blocks):
            norm = normalize_text(block)
            if not norm:
                continue
            h = text_to_hash(norm)
            items.append(
                Vacancy(
                    source="hh_txt",
                    source_file=fname,
                    idx_in_source=i,
                    text_raw=block,
                    text_norm=norm,
                    text_hash=h,
                )
            )

    return items


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_telegram_jsonl(jsonl_path: str) -> List[Vacancy]:
    """
    Подхватывает outputs/telegram/tg_vacancies.jsonl
    (создаётся твоим scripts/00_parse_telegram.py)
    """
    items: List[Vacancy] = []
    if not os.path.exists(jsonl_path):
        return items

    rows = load_jsonl(jsonl_path)

    for i, r in enumerate(rows):
        raw = r.get("text", "")
        norm = normalize_text(raw)
        if not norm:
            continue
        h = text_to_hash(norm)
        items.append(
            Vacancy(
                source="telegram_jsonl",
                source_file=os.path.basename(jsonl_path),
                idx_in_source=i,
                text_raw=raw,
                text_norm=norm,
                text_hash=h,
            )
        )

    return items


def exact_dedup(vacancies: List[Vacancy]) -> Tuple[List[Vacancy], pd.DataFrame]:
    """
    Одинаковый нормализованный текст -> дубликат
    """
    seen: Dict[str, int] = {}
    keep: List[Vacancy] = []
    drops = []

    for v in vacancies:
        if v.text_hash not in seen:
            seen[v.text_hash] = len(keep)
            keep.append(v)
        else:
            kept = keep[seen[v.text_hash]]
            drops.append(
                {
                    "type": "exact",
                    "dropped_source": v.source,
                    "dropped_source_file": v.source_file,
                    "dropped_idx": v.idx_in_source,
                    "kept_source": kept.source,
                    "kept_source_file": kept.source_file,
                    "kept_idx": kept.idx_in_source,
                    "similarity": 1.0,
                }
            )

    return keep, pd.DataFrame(drops)


def near_dedup_tfidf(
    vacancies: List[Vacancy],
    sim_threshold: float = 0.92,
    min_len_chars: int = 800,
) -> Tuple[List[Vacancy], pd.DataFrame]:
    """
    Near-dup:
    - TF-IDF (1-2 граммы)
    - cosine similarity
    - сравнение O(n^2), но для ~100-300 вакансий ок.

    sim_threshold:
      0.92 — умеренно “жёстко”. Если будет выкидывать “не то” — поднимем до 0.94-0.96.
    min_len_chars:
      чтобы не сравнивать короткие объявления (шум)
    """
    if not vacancies:
        return [], pd.DataFrame()

    texts = [v.text_norm for v in vacancies]
    lengths = [len(t) for t in texts]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        token_pattern=r"(?u)\b[\w\-\/\.]+\b",
    )
    X = vectorizer.fit_transform(texts)

    keep_mask = [True] * len(vacancies)
    drops = []

    for i in tqdm(range(len(vacancies)), desc="near-dedup"):
        if not keep_mask[i]:
            continue
        if lengths[i] < min_len_chars:
            continue

        sims = cosine_similarity(X[i], X).ravel()

        # проверяем только справа, чтобы не дублировать
        for j in range(i + 1, len(vacancies)):
            if not keep_mask[j]:
                continue
            if lengths[j] < min_len_chars:
                continue

            if sims[j] >= sim_threshold:
                keep_mask[j] = False
                drops.append(
                    {
                        "type": "near",
                        "dropped_source": vacancies[j].source,
                        "dropped_source_file": vacancies[j].source_file,
                        "dropped_idx": vacancies[j].idx_in_source,
                        "kept_source": vacancies[i].source,
                        "kept_source_file": vacancies[i].source_file,
                        "kept_idx": vacancies[i].idx_in_source,
                        "similarity": float(sims[j]),
                    }
                )

    keep = [v for v, m in zip(vacancies, keep_mask) if m]
    return keep, pd.DataFrame(drops)


def save_outputs(keep: List[Vacancy], drops_df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 1) единый JSONL датасет после дедупа
    jsonl_path = os.path.join(out_dir, "vacancies_dedup.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for v in keep:
            rec = {
                "source": v.source,
                "source_file": v.source_file,
                "idx_in_source": v.idx_in_source,
                "text": v.text_raw,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 2) отчёт о дублях
    drops_path = os.path.join(out_dir, "dedup_drops.csv")
    if drops_df is None or drops_df.empty:
        pd.DataFrame(
            columns=[
                "type",
                "dropped_source",
                "dropped_source_file",
                "dropped_idx",
                "kept_source",
                "kept_source_file",
                "kept_idx",
                "similarity",
            ]
        ).to_csv(drops_path, index=False, encoding="utf-8-sig")
    else:
        drops_df.to_csv(drops_path, index=False, encoding="utf-8-sig")

    print(f"[OK] kept={len(keep)}")
    print(f"[OK] drops={0 if drops_df is None else len(drops_df)}")
    print(f"[OK] wrote: {jsonl_path}")
    print(f"[OK] wrote: {drops_path}")


def main():
    raw_dir = os.path.join("data", "raw")
    out_dir = os.path.join("outputs", "dedup")

    # 1) HH/прочие txt
    vacancies = load_txt_files(raw_dir)

    # 2) Telegram jsonl (если есть)
    tg_jsonl = os.path.join("outputs", "telegram", "tg_vacancies.jsonl")
    vacancies += load_telegram_jsonl(tg_jsonl)

    print(f"[INFO] loaded={len(vacancies)} blocks (txt + telegram_jsonl)")

    # exact
    keep1, drops1 = exact_dedup(vacancies)
    print(f"[INFO] after exact keep={len(keep1)} drops={len(drops1)}")

    # near
    keep2, drops2 = near_dedup_tfidf(
        keep1,
        sim_threshold=0.92,
        min_len_chars=800,
    )
    print(f"[INFO] after near keep={len(keep2)} drops={len(drops2)}")

    drops = (
        pd.concat([drops1, drops2], ignore_index=True)
        if (drops1 is not None and not drops1.empty) or (drops2 is not None and not drops2.empty)
        else pd.DataFrame()
    )
    save_outputs(keep2, drops, out_dir)


if __name__ == "__main__":
    main()