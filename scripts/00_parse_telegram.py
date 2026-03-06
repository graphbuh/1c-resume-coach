import os
import re
import json
from typing import List

# 1) разделители/маркеры начала вакансии (простые и понятные)
START_PATTERNS = [
    r"^\s*вакансия\s*[:\-]",          # "Вакансия: ..."
    r"^\s*позиция\s*[:\-]",           # "Позиция: ..."
    r"^\s*роль\s*[:\-]",              # "Роль: ..."
    r"^\s*#вакансия\b",               # "#вакансия ..."
    r"^\s*ищем\b",                    # "Ищем ..."
    r"^\s*приглашаем\b",              # "Приглашаем ..."
]

START_RE = re.compile("|".join(START_PATTERNS), flags=re.IGNORECASE | re.MULTILINE)

# 2) штуки, которые будем обезличивать/вырезать (контакты)
RE_PHONE = re.compile(r"(\+7|8)\s*[\(\- ]?\d{3}[\)\- ]?\s*\d{3}[\- ]?\d{2}[\- ]?\d{2}")
RE_TG = re.compile(r"@\w+")
RE_EMAIL = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")

# 3) хэштеги — часто шум, но иногда по ним можно понимать роль (#аналитик1С)
# поэтому: выкинем строку целиком только если она состоит в основном из хэштегов
RE_MOSTLY_HASHTAGS = re.compile(r"^\s*(#\w+\s*){2,}$", flags=re.IGNORECASE)

def normalize_newlines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def anonymize_contacts(text: str) -> str:
    text = RE_PHONE.sub("[PHONE]", text)
    text = RE_EMAIL.sub("[EMAIL]", text)
    text = RE_TG.sub("[TG]", text)
    return text

def light_cleanup_lines(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        l = line.strip()
        if not l:
            lines.append("")
            continue

        # выкидываем “служебные” строки, где почти одни хэштеги
        if RE_MOSTLY_HASHTAGS.match(l):
            continue

        # иногда встречается "Благодарим за публикацию..." — тоже шум
        if "благодарим за публикацию" in l.lower():
            continue

        lines.append(l)

    cleaned = "\n".join(lines)
    cleaned = normalize_newlines(cleaned)
    return cleaned

def split_telegram_posts(text: str) -> List[str]:
    text = normalize_newlines(text)

    # Способ 1: режем по "стартам" вакансий
    # Делается так: находим все позиции START_RE и режем между ними.
    starts = [m.start() for m in START_RE.finditer(text)]
    if not starts:
        # fallback: режем по двойным пустым строкам
        parts = re.split(r"\n\s*\n\s*\n+", text)
        return [p.strip() for p in parts if p.strip()]

    parts = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        chunk = text[s:e].strip()
        if chunk:
            parts.append(chunk)
    return parts

def main():
    in_path = os.path.join("data", "raw", "Вакансии.txt")
    out_dir = os.path.join("outputs", "telegram")
    os.makedirs(out_dir, exist_ok=True)

    with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    posts = split_telegram_posts(raw)

    out_path = os.path.join(out_dir, "tg_vacancies.jsonl")
    kept = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, p in enumerate(posts):
            cleaned = light_cleanup_lines(p)
            cleaned = anonymize_contacts(cleaned)

            # фильтр “слишком короткое”
            if len(cleaned) < 200:
                continue

            rec = {
                "source": "telegram",
                "source_file": "Вакансии.txt",
                "idx": idx,
                "text": cleaned,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[OK] telegram blocks total={len(posts)} kept={kept}")
    print(f"[OK] wrote: {out_path}")

if __name__ == "__main__":
    main()