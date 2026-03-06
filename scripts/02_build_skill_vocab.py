import os
import re
import json
from collections import Counter, defaultdict

import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer


# -----------------------------
# Paths
# -----------------------------
DEDUP_JSONL = os.path.join("outputs", "dedup", "vacancies_dedup.jsonl")
VOCAB_DIR = os.path.join("outputs", "vocab")
SEED_YAML = os.path.join(VOCAB_DIR, "skills_seed.yaml")

OUT_KNOWN = os.path.join(VOCAB_DIR, "known_skills_stat.csv")
OUT_UNKNOWN = os.path.join(VOCAB_DIR, "unknown_terms.csv")
OUT_TOP_TERMS = os.path.join(VOCAB_DIR, "top_terms_all.csv")
OUT_META = os.path.join(VOCAB_DIR, "run_meta.json")


# -----------------------------
# Text normalization
# -----------------------------
def normalize_text_basic(s: str) -> str:
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


def strip_obvious_noise(s: str) -> str:
    """
    Чистим типовой мусор, который часто мешает TF-IDF:
    - телефоны/email/tg
    - зарплаты, график, ТК РФ (не полностью, но подсечём)
    """
    if not s:
        return ""

    # контакты
    s = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", " [EMAIL] ", s)
    s = re.sub(r"@\w+", " [TG] ", s)
    s = re.sub(r"(\+7|8)\s*[\(\- ]?\d{3}[\)\- ]?\s*\d{3}[\- ]?\d{2}[\- ]?\d{2}", " [PHONE] ", s)

    # денежные суммы / валюты
    s = re.sub(r"\b\d{2,3}\s?000\b", " [MONEY] ", s)  # 120 000
    s = re.sub(r"\b\d+\s*(₽|руб\.?|рублей|р\.)\b", " [MONEY] ", s)

    # типовые фразы-«шум»
    # (оставляем мягко, чтобы не сломать смысл)
    noise_phrases = [
        "оформление по тк", "по тк рф", "соц пакет", "социальный пакет",
        "график 5/2", "график 5 2", "полная занятость", "удаленная работа",
        "формат работы", "выплаты два раза", "до вычета налогов", "на руки",
    ]
    for p in noise_phrases:
        s = s.replace(p, " ")

    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


# -----------------------------
# Seed YAML helpers
# -----------------------------
def ensure_seed_yaml_exists(path: str) -> None:
    """
    Если seed-словаря нет — создаём минимальный шаблон.
    Его можно потом расширять вручную (или с моей помощью).
    """
    if os.path.exists(path):
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    template = {
        "core": {
            "language_1c": {"aliases": ["программирование 1с", "язык 1с", "встроенный язык"]},
            "skd": {"aliases": ["скд", "система компоновки данных"]},
            "queries": {"aliases": ["язык запросов", "запросы 1с", "запросы"]},
            "managed_forms": {"aliases": ["управляемые формы", "управляемое приложение"]},
            "bsp": {"aliases": ["бсп", "стандартная библиотека подсистем", "стандартные подсистемы"]},
        },
        "integrations": {
            "http_rest": {"aliases": ["rest", "rest api", "http-сервисы", "api", "http"]},
            "soap": {"aliases": ["soap", "wsdl"]},
            "odata": {"aliases": ["odata"]},
            "exchange_kd": {"aliases": ["кд2", "кд 2", "кд3", "кд 3", "конвертация данных", "обмен"]},
            "kafka": {"aliases": ["kafka"]},
            "rabbitmq": {"aliases": ["rabbitmq", "rmq"]},
        },
        "sql_perf": {
            "postgresql": {"aliases": ["postgres", "postgresql"]},
            "mssql": {"aliases": ["mssql", "ms sql", "sql server"]},
            "tech_log": {"aliases": ["технологический журнал", "техжурнал"]},
            "optimization": {"aliases": ["оптимизация", "производительность", "высоконагруж", "highload"]},
        },
        "configs": {
            "erp": {"aliases": ["erp", "1с erp", "1с:erp"]},
            "ut": {"aliases": ["ут", "ут 11", "управление торговлей"]},
            "zup": {"aliases": ["зуп", "зарплата и управление персоналом"]},
            "bp": {"aliases": ["бп", "бухгалтерия", "1с бухгалтерия", "бухгалтерия предприятия"]},
            "do": {"aliases": ["документооборот", "1с до", "1с:до"]},
            "ka": {"aliases": ["ка", "комплексная автоматизация"]},
            "upp": {"aliases": ["упп", "управление производственным предприятием"]},
            "uh": {"aliases": ["ух", "управление холдингом"]},
            "bgu": {"aliases": ["бгу", "бухгалтерия государственного учреждения"]},
        },
        "tools_process": {
            "git": {"aliases": ["git", "gitlab", "github"]},
            "edt": {"aliases": ["edt", "enterprise development tools"]},
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(template, f, allow_unicode=True, sort_keys=False)

    print(f"[INFO] Seed file was missing. Created template: {path}")


def load_seed(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def flatten_seed_aliases(seed: dict) -> dict:
    """
    Возвращает alias_map:
      normalized_alias -> (category, skill_key)
    """
    alias_map = {}
    for cat, skills in seed.items():
        if not isinstance(skills, dict):
            continue
        for skill_key, spec in skills.items():
            if not isinstance(spec, dict):
                continue
            aliases = spec.get("aliases", [])
            if not aliases:
                continue
            for a in aliases:
                aa = normalize_text_basic(str(a))
                if aa:
                    alias_map[aa] = (cat, skill_key)
    return alias_map


def extract_known_skills(text: str, alias_map: dict) -> set[tuple[str, str]]:
    """
    Простая эвристика: если alias встречается подстрокой — считаем навык найденным.
    """
    t = normalize_text_basic(text)
    found = set()
    for alias, (cat, skill_key) in alias_map.items():
        if alias and alias in t:
            found.add((cat, skill_key))
    return found


# -----------------------------
# TF-IDF candidates
# -----------------------------
RU_STOPWORDS_LIGHT = set("""
и в во на по к ко из от до для при с со а но или что это как так же уже еще
мы вы ты он она они вы будете будет есть нет да
опыт работы требования обязанности условия график оформление компания
удаленно удаленная удаленный офис гибрид полный день занятость
руб рубль рублей р р.
""".split())


def looks_like_noise_term(term: str) -> bool:
    """
    Фильтр мусора:
    - только цифры
    - слишком короткое
    - состоит из стоп-слов
    - типовые HR слова
    """
    t = term.strip().lower()

    if len(t) < 3:
        return True
    if re.fullmatch(r"\d+", t):
        return True
    if t in RU_STOPWORDS_LIGHT:
        return True

    # слишком "HR"
    hr_bad = [
        "оформление", "тк", "трудовой", "занятость", "график", "премии",
        "зарплата", "выплаты", "компенсация", "соцпакет", "дмс", "корпоратив",
        "отпуск", "больничный",
    ]
    if any(x == t for x in hr_bad):
        return True

    # похож на мусорные шаблоны
    if re.search(r"\b(5/2|52)\b", t):
        return True

    return False


def build_tfidf_candidates(texts: list[str], top_k: int = 800) -> pd.DataFrame:
    """
    Строим кандидатов терминов/фраз 1..3 слов по TF-IDF.
    Важно: это "термин-кандидат", дальше будет ручная чистка.
    """
    vect = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=2,          # встречается минимум в 2 вакансиях
        max_df=0.9,        # если в >90% вакансий - слишком общее
        token_pattern=r"(?u)\b[\w\-\/\.]+\b",
    )
    X = vect.fit_transform(texts)
    terms = vect.get_feature_names_out()

    # средняя важность по корпусу
    scores = X.mean(axis=0).A1
    df = pd.DataFrame({"term": terms, "score_mean_tfidf": scores})
    df = df.sort_values("score_mean_tfidf", ascending=False).head(top_k).reset_index(drop=True)

    # дополнительные "простые" поля
    df["len_chars"] = df["term"].str.len()
    df["is_noise"] = df["term"].apply(looks_like_noise_term)

    return df


# -----------------------------
# JSONL loader
# -----------------------------
def load_dedup_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(VOCAB_DIR, exist_ok=True)

    # 0) seed file
    ensure_seed_yaml_exists(SEED_YAML)
    seed = load_seed(SEED_YAML)
    alias_map = flatten_seed_aliases(seed)

    # 1) load corpus
    rows = load_dedup_jsonl(DEDUP_JSONL)

    texts = []
    sources = Counter()
    for r in rows:
        sources[r.get("source", "unknown")] += 1

        t = r.get("text", "")
        t = normalize_text_basic(t)
        t = strip_obvious_noise(t)
        if len(t) < 120:
            continue
        texts.append(t)

    print(f"[INFO] loaded vacancies_dedup rows={len(rows)}")
    print(f"[INFO] usable texts={len(texts)}")
    print(f"[INFO] sources breakdown: {dict(sources)}")

    # 2) known skill stats from seed
    skill_counter = Counter()
    for t in texts:
        found = extract_known_skills(t, alias_map)
        for cat, key in found:
            skill_counter[(cat, key)] += 1

    known_df = pd.DataFrame(
        [{"category": c, "skill": k, "count": n} for (c, k), n in skill_counter.most_common()]
    )
    known_df.to_csv(OUT_KNOWN, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote: {OUT_KNOWN} (rows={len(known_df)})")

    # 3) tf-idf candidates
    cand_df = build_tfidf_candidates(texts, top_k=800)
    cand_df.to_csv(OUT_TOP_TERMS, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote: {OUT_TOP_TERMS} (rows={len(cand_df)})")

    # 4) unknown terms: remove those already covered by seed aliases (точным совпадением алиаса)
    seed_aliases_set = set(alias_map.keys())

    def is_covered_by_seed(term: str) -> bool:
        return normalize_text_basic(term) in seed_aliases_set

    unknown = cand_df.copy()
    unknown["covered_by_seed_alias_exact"] = unknown["term"].apply(is_covered_by_seed)

    # фильтруем мусор и покрытые
    unknown = unknown[~unknown["is_noise"]].copy()
    unknown = unknown[~unknown["covered_by_seed_alias_exact"]].copy()

    # дополнительный фильтр: убрать прям совсем общие слова
    unknown = unknown[unknown["len_chars"] >= 3].copy()
    unknown = unknown.reset_index(drop=True)

    unknown.to_csv(OUT_UNKNOWN, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote: {OUT_UNKNOWN} (rows={len(unknown)})")

    # 5) meta
    meta = {
        "dedup_jsonl": DEDUP_JSONL,
        "total_rows_in_dedup": len(rows),
        "usable_texts": len(texts),
        "sources": dict(sources),
        "seed_yaml": SEED_YAML,
        "seed_alias_count": len(seed_aliases_set),
        "outputs": {
            "known_skills_stat": OUT_KNOWN,
            "top_terms_all": OUT_TOP_TERMS,
            "unknown_terms": OUT_UNKNOWN,
        },
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote: {OUT_META}")


if __name__ == "__main__":
    main()