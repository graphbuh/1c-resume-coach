import os
import re
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import yaml


DEDUP_JSONL = os.path.join("outputs", "dedup", "vacancies_dedup.jsonl")
SKILLS_YAML = os.path.join("outputs", "vocab", "skills.yaml")

OUT_DIR = os.path.join("outputs", "skills")
OUT_CSV = os.path.join(OUT_DIR, "vacancy_skills.csv")
OUT_JSONL = os.path.join(OUT_DIR, "vacancy_skills.jsonl")
OUT_FREQ = os.path.join(OUT_DIR, "skill_frequency.csv")
OUT_META = os.path.join(OUT_DIR, "run_meta.json")


def normalize_text(s: str) -> str:
    """Нормализация текста для поиска навыков."""
    if not s:
        return ""
    s = s.replace("\u00ad", "")  # soft hyphen
    s = s.replace("ё", "е").replace("Ё", "Е")
    s = s.lower()
    s = re.sub(r"\b1c\b", "1с", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def load_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_skills_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Skills YAML not found: {path}\n"
            f"Create it first (we generated skills.yaml earlier)."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("skills.yaml must be a mapping of categories -> skills")
    return data


def build_alias_index(skills: dict) -> Tuple[Dict[str, Tuple[str, str]], Dict[Tuple[str, str], List[str]]]:
    """
    Возвращает:
      alias_to_skill: normalized_alias -> (category, skill_key)
      skill_to_aliases: (category, skill_key) -> [aliases...]
    """
    alias_to_skill: Dict[str, Tuple[str, str]] = {}
    skill_to_aliases: Dict[Tuple[str, str], List[str]] = {}

    for category, skill_map in skills.items():
        if not isinstance(skill_map, dict):
            continue
        for skill_key, spec in skill_map.items():
            if not isinstance(spec, dict):
                continue
            aliases = spec.get("aliases", [])
            if not aliases:
                continue

            norm_aliases = []
            for a in aliases:
                aa = normalize_text(str(a))
                if not aa:
                    continue
                norm_aliases.append(aa)
                # если алиас повторяется в разных навыках — оставим первый (обычно лучше не допускать)
                alias_to_skill.setdefault(aa, (category, skill_key))

            skill_to_aliases[(category, skill_key)] = norm_aliases

    return alias_to_skill, skill_to_aliases


def extract_skills_from_text(
    text_norm: str,
    skill_to_aliases: Dict[Tuple[str, str], List[str]]
) -> List[Dict[str, str]]:
    """
    Возвращает список найденных навыков в виде:
      [{"category":..., "skill":..., "matched_alias":...}, ...]
    """
    found = []
    for (cat, skill), aliases in skill_to_aliases.items():
        for a in aliases:
            if a and a in text_norm:
                found.append({"category": cat, "skill": skill, "matched_alias": a})
                break  # один навык = один матч достаточно
    return found


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) load data
    vacancies = load_jsonl(DEDUP_JSONL)
    skills = load_skills_yaml(SKILLS_YAML)

    alias_to_skill, skill_to_aliases = build_alias_index(skills)

    print(f"[INFO] loaded vacancies: {len(vacancies)} from {DEDUP_JSONL}")
    print(f"[INFO] loaded skill categories: {len(skills)} from {SKILLS_YAML}")
    print(f"[INFO] total skills: {len(skill_to_aliases)}")
    print(f"[INFO] total aliases: {len(alias_to_skill)}")

    # 2) extract
    rows_csv = []
    rows_jsonl = []
    freq = Counter()
    freq_by_category = Counter()

    for vidx, v in enumerate(vacancies):
        source = v.get("source", "")
        source_file = v.get("source_file", "")
        idx_in_source = v.get("idx_in_source", None)
        text_raw = v.get("text", "")

        text_norm = normalize_text(text_raw)
        found = extract_skills_from_text(text_norm, skill_to_aliases)

        # сводные поля
        found_pairs = [(x["category"], x["skill"]) for x in found]
        for cat, skill in found_pairs:
            freq[(cat, skill)] += 1
            freq_by_category[cat] += 1

        # для CSV: одна строка на вакансию
        unique_skill_names = [f'{x["category"]}:{x["skill"]}' for x in found]
        unique_skill_names = sorted(set(unique_skill_names))

        rows_csv.append(
            {
                "vacancy_id": vidx,
                "source": source,
                "source_file": source_file,
                "idx_in_source": idx_in_source,
                "text_len": len(text_raw),
                "skills_count": len(unique_skill_names),
                "skills": "; ".join(unique_skill_names),
            }
        )

        # для JSONL: подробная запись
        rows_jsonl.append(
            {
                "vacancy_id": vidx,
                "source": source,
                "source_file": source_file,
                "idx_in_source": idx_in_source,
                "skills": found,  # с matched_alias
            }
        )

    # 3) write outputs
    df = pd.DataFrame(rows_csv)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in rows_jsonl:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    freq_rows = []
    for (cat, skill), c in freq.most_common():
        freq_rows.append({"category": cat, "skill": skill, "count": c})
    freq_df = pd.DataFrame(freq_rows)
    freq_df.to_csv(OUT_FREQ, index=False, encoding="utf-8-sig")

    meta = {
        "dedup_jsonl": DEDUP_JSONL,
        "skills_yaml": SKILLS_YAML,
        "vacancies": len(vacancies),
        "skills_total": len(skill_to_aliases),
        "aliases_total": len(alias_to_skill),
        "outputs": {
            "vacancy_skills_csv": OUT_CSV,
            "vacancy_skills_jsonl": OUT_JSONL,
            "skill_frequency_csv": OUT_FREQ,
        },
        "category_counts": dict(freq_by_category),
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote: {OUT_CSV}")
    print(f"[OK] wrote: {OUT_JSONL}")
    print(f"[OK] wrote: {OUT_FREQ}")
    print(f"[OK] wrote: {OUT_META}")


if __name__ == "__main__":
    main()