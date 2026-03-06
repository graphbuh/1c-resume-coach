import os
import json
from collections import Counter

import pandas as pd


IN_CSV = os.path.join("outputs", "skills", "vacancy_skills.csv")

OUT_DIR = os.path.join("outputs", "grades")
OUT_GRADES = os.path.join(OUT_DIR, "vacancy_grades.csv")
OUT_DIST = os.path.join(OUT_DIR, "grade_distribution.csv")
OUT_META = os.path.join(OUT_DIR, "run_meta.json")


def parse_skills_cell(cell: str) -> set[str]:
    """
    skills cell format: "category:skill; category:skill; ..."
    """
    if not isinstance(cell, str) or not cell.strip():
        return set()
    parts = [p.strip() for p in cell.split(";")]
    parts = [p for p in parts if p]
    return set(parts)


def has_any(skills: set[str], keys: list[str]) -> bool:
    return any(k in skills for k in keys)


def count_any(skills: set[str], keys: list[str]) -> int:
    return sum(1 for k in keys if k in skills)


def grade_scores(skills: set[str], text_len: int) -> dict:
    """
    Возвращает скоринги по грейдам.
    Правила подобраны для 1С-рынка:
    - intern/junior: сопровождение/обновления, типовые конфиги, базовый стек
    - middle: интеграции/обмены, права, БСП, оптимизация
    - senior: производительность, техжурнал, блокировки, сложные интеграции + архитектурные элементы
    - architect: архитектура/интеграционная архитектура/техлид/наставничество
    """
    # ключевые наборы сигналов
    junior_signals = [
        "dev_language_and_metadata:updates_support",
        "analysis_consulting_process:user_training",
        "queries_reports_ui:reports",
        "queries_reports_ui:queries",
        "queries_reports_ui:skd",
        "core_platform:managed_application",
        "dev_language_and_metadata:language_1c",
    ]

    middle_signals = [
        "integrations_and_exchange:exchange_kd",
        "integrations_and_exchange:http_rest",
        "integrations_and_exchange:integration_general",
        "dev_language_and_metadata:roles_and_rights",
        "core_platform:bsp",
        "sql_perf_admin:sql",
        "sql_perf_admin:optimization",
        "queries_reports_ui:query_optimization",
        "tools_process:git",
    ]

    senior_signals = [
        "sql_perf_admin:tech_log",
        "sql_perf_admin:locks_transactions",
        "sql_perf_admin:optimization",
        "integrations_and_exchange:message_brokers",
        "architecture_leadership:mentoring_review",
        "dev_language_and_metadata:code_quality",
    ]

    architect_signals = [
        "architecture_leadership:architecture",
        "architecture_leadership:integration_architecture",
        "architecture_leadership:tech_lead",
        "architecture_leadership:pre_project",
        "architecture_leadership:mentoring_review",
    ]

    # конфиги как контекст (сами по себе не делают грейд, но усиливают)
    config_hits = count_any(skills, [
        "configs_typical:erp",
        "configs_typical:uh",
        "configs_typical:ka",
        "configs_typical:ut",
        "configs_typical:zup",
        "configs_typical:bp",
        "configs_typical:do",
        "configs_typical:upp",
        "configs_typical:bgu",
        "configs_typical:wms_tms",
    ])

    # базовые очки
    score_intern = 0.0
    score_junior = 0.0
    score_middle = 0.0
    score_senior = 0.0
    score_architect = 0.0

    # intern: очень мало сигналов, короткие тексты, часто только обучение/поддержка
    # (мы intern используем аккуратно, это скорее "низкий junior")
    if text_len < 900:
        score_intern += 0.5
    score_intern += 0.3 * count_any(skills, [
        "analysis_consulting_process:user_training",
        "dev_language_and_metadata:updates_support",
        "analysis_consulting_process:documentation",
    ])
    if config_hits >= 1:
        score_intern += 0.2

    # junior
    score_junior += 0.25 * count_any(skills, junior_signals)
    score_junior += 0.08 * min(config_hits, 4)

    # middle
    score_middle += 0.30 * count_any(skills, middle_signals)
    score_middle += 0.10 * min(config_hits, 5)

    # senior
    score_senior += 0.45 * count_any(skills, senior_signals)
    score_senior += 0.08 * min(config_hits, 6)

    # architect
    score_architect += 0.55 * count_any(skills, architect_signals)
    score_architect += 0.05 * min(config_hits, 6)

    # усиление: если есть архитектура и техлид — точно architect
    if has_any(skills, ["architecture_leadership:tech_lead"]) and has_any(skills, ["architecture_leadership:architecture"]):
        score_architect += 1.0

    # усиление: брокеры сообщений + техжурнал/блокировки → senior+
    if has_any(skills, ["integrations_and_exchange:message_brokers"]) and has_any(skills, ["sql_perf_admin:tech_log", "sql_perf_admin:locks_transactions"]):
        score_senior += 0.8

    # защита от "слишком общего" консультанта:
    # если много процессов (requirements, consulting, training), но нет тех сигналов — это не middle dev
    process_hits = count_any(skills, [
        "analysis_consulting_process:requirements",
        "analysis_consulting_process:consulting",
        "analysis_consulting_process:user_training",
        "analysis_consulting_process:implementation",
        "analysis_consulting_process:tz_spec",
        "analysis_consulting_process:business_processes",
    ])
    tech_hits = count_any(skills, middle_signals + senior_signals + architect_signals)
    if process_hits >= 3 and tech_hits == 0:
        score_middle -= 0.4
        score_senior -= 0.6

    return {
        "intern": round(score_intern, 4),
        "junior": round(score_junior, 4),
        "middle": round(score_middle, 4),
        "senior": round(score_senior, 4),
        "architect": round(score_architect, 4),
        "config_hits": int(config_hits),
        "process_hits": int(process_hits),
        "tech_hits": int(tech_hits),
    }


def pick_grade(scores: dict) -> tuple[str, float]:
    grade_keys = ["architect", "senior", "middle", "junior", "intern"]
    best_g = None
    best_s = -1e9
    for g in grade_keys:
        s = scores[g]
        if s > best_s:
            best_s = s
            best_g = g

    # минимальные пороги, чтобы не ставить "architect" по одному слову
    if best_g == "architect" and best_s < 1.0:
        # если архитектурный скоринг слабый — уступим senior
        best_g = "senior"
        best_s = scores["senior"]

    if best_g == "senior" and best_s < 0.9:
        best_g = "middle"
        best_s = scores["middle"]

    if best_g == "middle" and best_s < 0.8:
        best_g = "junior"
        best_s = scores["junior"]

    if best_g == "junior" and best_s < 0.6:
        best_g = "intern"
        best_s = scores["intern"]

    return best_g, float(best_s)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(f"Input not found: {IN_CSV}\nRun 03_extract_skills.py first.")

    df = pd.read_csv(IN_CSV, encoding="utf-8-sig")
    rows = []
    dist = Counter()

    for _, r in df.iterrows():
        skills = parse_skills_cell(r.get("skills", ""))
        text_len = int(r.get("text_len", 0))

        scores = grade_scores(skills, text_len)
        grade, confidence = pick_grade(scores)

        dist[grade] += 1

        rows.append({
            "vacancy_id": int(r["vacancy_id"]),
            "source": r.get("source", ""),
            "source_file": r.get("source_file", ""),
            "idx_in_source": r.get("idx_in_source", ""),
            "text_len": text_len,
            "skills_count": int(r.get("skills_count", 0)),
            "grade": grade,
            "confidence": round(confidence, 4),
            "score_intern": scores["intern"],
            "score_junior": scores["junior"],
            "score_middle": scores["middle"],
            "score_senior": scores["senior"],
            "score_architect": scores["architect"],
            "config_hits": scores["config_hits"],
            "process_hits": scores["process_hits"],
            "tech_hits": scores["tech_hits"],
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_GRADES, index=False, encoding="utf-8-sig")

    dist_df = pd.DataFrame([{"grade": g, "count": c} for g, c in dist.most_common()])
    dist_df.to_csv(OUT_DIST, index=False, encoding="utf-8-sig")

    meta = {
        "input_csv": IN_CSV,
        "rows": int(len(df)),
        "outputs": {
            "vacancy_grades": OUT_GRADES,
            "grade_distribution": OUT_DIST,
        },
        "distribution": dict(dist),
    }
    with open(OUT_META, "w", encoding="utf-8", newline="") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote: {OUT_GRADES}")
    print(f"[OK] wrote: {OUT_DIST}")
    print(f"[OK] wrote: {OUT_META}")
    print(f"[INFO] grade distribution: {dict(dist)}")


if __name__ == "__main__":
    main()