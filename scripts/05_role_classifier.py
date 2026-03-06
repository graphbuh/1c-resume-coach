import os
import json
from collections import Counter

import pandas as pd


IN_CSV = os.path.join("outputs", "skills", "vacancy_skills.csv")

OUT_DIR = os.path.join("outputs", "roles")
OUT_ROLES = os.path.join(OUT_DIR, "vacancy_roles.csv")
OUT_DIST = os.path.join(OUT_DIR, "role_distribution.csv")
OUT_META = os.path.join(OUT_DIR, "run_meta.json")


def parse_skills_cell(cell: str) -> set[str]:
    """
    skills cell format:
      "category:skill; category:skill; ..."
    """
    if not isinstance(cell, str) or not cell.strip():
        return set()

    parts = [p.strip() for p in cell.split(";")]
    return {p for p in parts if p}


def count_hits(skills: set[str], keys: list[str]) -> int:
    return sum(1 for k in keys if k in skills)


def role_scores(skills: set[str]) -> dict:
    """
    Возвращает сырые скоры ролей.
    """

    programmer_keys = [
        "dev_language_and_metadata:language_1c",
        "queries_reports_ui:queries",
        "queries_reports_ui:skd",
        "queries_reports_ui:query_optimization",
        "dev_language_and_metadata:common_modules",
        "dev_language_and_metadata:event_handlers",
        "core_platform:extensions_support",
        "core_platform:bsp",
        "dev_language_and_metadata:code_quality",
        "sql_perf_admin:sql",
        "sql_perf_admin:optimization",
        "sql_perf_admin:locks_transactions",
        "integrations_and_exchange:http_rest",
        "integrations_and_exchange:exchange_kd",
        "integrations_and_exchange:integration_general",
        "integrations_and_exchange:message_brokers",
        "tools_process:git",
        "tools_process:ci_cd",
    ]

    analyst_keys = [
        "analysis_consulting_process:requirements",
        "analysis_consulting_process:tz_spec",
        "analysis_consulting_process:business_processes",
        "analysis_consulting_process:documentation",
        "analysis_consulting_process:testing_uat",
        "analysis_consulting_process:governance",
        "architecture_functional_signals:process_modeling",
        "architecture_functional_signals:requirements_governance",
    ]

    consultant_keys = [
        "analysis_consulting_process:consulting",
        "analysis_consulting_process:user_training",
        "analysis_consulting_process:implementation",
        "analysis_consulting_process:documentation",
        "domain_functional:regulated_accounting",
        "domain_functional:finance_accounting",
        "domain_functional:hr_payroll",
        "configs_typical:bp",
        "configs_typical:zup",
        "configs_typical:erp",
        "configs_typical:ut",
        "configs_typical:ka",
        "configs_typical:uh",
        "configs_typical:do",
        "configs_typical:bgu",
    ]

    architect_technical_keys = [
        "architecture_roles:technical_architect",
        "architecture_technical_signals:technical_architecture",
        "architecture_technical_signals:integration_architecture",
        "architecture_technical_signals:performance_architecture",
        "architecture_technical_signals:devops_quality_stack",
        "leadership_general:tech_lead",
        "leadership_general:mentoring_review",
        "leadership_general:pre_project",
        "sql_perf_admin:tech_log",
        "sql_perf_admin:optimization",
        "integrations_and_exchange:bus_esb",
        "integrations_and_exchange:message_brokers",
    ]

    architect_functional_keys = [
        "architecture_roles:functional_architect",
        "architecture_functional_signals:functional_architecture",
        "architecture_functional_signals:process_modeling",
        "architecture_functional_signals:requirements_governance",
        "architecture_functional_signals:solution_governance",
        "analysis_consulting_process:governance",
        "analysis_consulting_process:requirements",
        "analysis_consulting_process:business_processes",
        "analysis_consulting_process:documentation",
        "leadership_general:pre_project",
    ]

    programmer_score = 0.0
    analyst_score = 0.0
    consultant_score = 0.0
    arch_tech_score = 0.0
    arch_func_score = 0.0

    programmer_score += 0.30 * count_hits(skills, programmer_keys)
    analyst_score += 0.35 * count_hits(skills, analyst_keys)
    consultant_score += 0.28 * count_hits(skills, consultant_keys)
    arch_tech_score += 0.45 * count_hits(skills, architect_technical_keys)
    arch_func_score += 0.45 * count_hits(skills, architect_functional_keys)

    # Усиления для программиста
    if (
        "dev_language_and_metadata:language_1c" in skills
        and "queries_reports_ui:queries" in skills
    ):
        programmer_score += 0.8

    if (
        "integrations_and_exchange:http_rest" in skills
        or "integrations_and_exchange:exchange_kd" in skills
    ):
        programmer_score += 0.4

    # Усиления для аналитика
    if (
        "analysis_consulting_process:requirements" in skills
        and "analysis_consulting_process:tz_spec" in skills
    ):
        analyst_score += 0.8

    if (
        "analysis_consulting_process:business_processes" in skills
        and "analysis_consulting_process:documentation" in skills
    ):
        analyst_score += 0.6

    # Усиления для консультанта
    if (
        "analysis_consulting_process:consulting" in skills
        and "analysis_consulting_process:user_training" in skills
    ):
        consultant_score += 0.8

    if "domain_functional:regulated_accounting" in skills:
        consultant_score += 0.4

    # Усиления для технического архитектора
    if "architecture_roles:technical_architect" in skills:
        arch_tech_score += 2.0

    if (
        "architecture_technical_signals:technical_architecture" in skills
        and "architecture_technical_signals:integration_architecture" in skills
    ):
        arch_tech_score += 1.5

    if (
        "architecture_technical_signals:performance_architecture" in skills
        and "leadership_general:tech_lead" in skills
    ):
        arch_tech_score += 1.2

    # Усиления для функционального архитектора
    if "architecture_roles:functional_architect" in skills:
        arch_func_score += 2.0

    if (
        "architecture_functional_signals:functional_architecture" in skills
        and "architecture_functional_signals:process_modeling" in skills
    ):
        arch_func_score += 1.5

    if (
        "architecture_functional_signals:requirements_governance" in skills
        and "architecture_functional_signals:solution_governance" in skills
    ):
        arch_func_score += 1.2

    # Немного штрафуем "программиста", если текст чисто процессный
    process_only_hits = count_hits(
        skills,
        [
            "analysis_consulting_process:requirements",
            "analysis_consulting_process:business_processes",
            "analysis_consulting_process:user_training",
            "analysis_consulting_process:consulting",
            "analysis_consulting_process:documentation",
        ],
    )
    tech_only_hits = count_hits(
        skills,
        [
            "dev_language_and_metadata:language_1c",
            "queries_reports_ui:queries",
            "queries_reports_ui:skd",
            "sql_perf_admin:sql",
            "integrations_and_exchange:http_rest",
            "integrations_and_exchange:exchange_kd",
            "core_platform:bsp",
        ],
    )

    if process_only_hits >= 4 and tech_only_hits == 0:
        programmer_score -= 0.7

    return {
        "programmer": round(programmer_score, 4),
        "analyst": round(analyst_score, 4),
        "consultant": round(consultant_score, 4),
        "architect_technical": round(arch_tech_score, 4),
        "architect_functional": round(arch_func_score, 4),
    }


def choose_role(scores: dict) -> tuple[str, float, str]:
    """
    Возвращает:
      role, confidence, note
    """
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    top_role, top_score = ordered[0]
    second_role, second_score = ordered[1]

    pair = {top_role, second_role}
    gap = top_score - second_score

    # Приоритет архитекторов
    if top_role in {"architect_technical", "architect_functional"} and top_score >= 1.2:
        return top_role, float(top_score), "direct_architect_match"

    hybrid_pairs = [
        {"analyst", "consultant"},
        {"analyst", "programmer"},
        {"consultant", "programmer"},
    ]

    if pair in hybrid_pairs and second_score >= 1.0 and gap <= 0.6:
        ordered_pair = sorted([top_role, second_role])
        hybrid_name = f"hybrid_{ordered_pair[0]}_{ordered_pair[1]}"
        return hybrid_name, float(top_score), "hybrid_close_scores"

    # Если рядом архитектор — даем ему приоритет
    if "architect_technical" in pair and scores["architect_technical"] >= 1.0:
        return "architect_technical", float(scores["architect_technical"]), "architect_priority"

    if "architect_functional" in pair and scores["architect_functional"] >= 1.0:
        return "architect_functional", float(scores["architect_functional"]), "architect_priority"

    return top_role, float(top_score), "top_score"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(f"Input not found: {IN_CSV}\nRun 03_extract_skills.py first.")

    df = pd.read_csv(IN_CSV, encoding="utf-8-sig")
    rows = []
    dist = Counter()

    for _, r in df.iterrows():
        skills = parse_skills_cell(r.get("skills", ""))
        scores = role_scores(skills)
        role, confidence, note = choose_role(scores)

        dist[role] += 1

        rows.append(
            {
                "vacancy_id": int(r["vacancy_id"]),
                "source": r.get("source", ""),
                "source_file": r.get("source_file", ""),
                "idx_in_source": r.get("idx_in_source", ""),
                "text_len": int(r.get("text_len", 0)),
                "skills_count": int(r.get("skills_count", 0)),
                "role": role,
                "confidence": round(confidence, 4),
                "decision_note": note,
                "score_programmer": scores["programmer"],
                "score_analyst": scores["analyst"],
                "score_consultant": scores["consultant"],
                "score_architect_technical": scores["architect_technical"],
                "score_architect_functional": scores["architect_functional"],
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_ROLES, index=False, encoding="utf-8-sig")

    dist_df = pd.DataFrame([{"role": k, "count": v} for k, v in dist.most_common()])
    dist_df.to_csv(OUT_DIST, index=False, encoding="utf-8-sig")

    meta = {
        "input_csv": IN_CSV,
        "rows": int(len(df)),
        "distribution": dict(dist),
        "outputs": {
            "vacancy_roles_csv": OUT_ROLES,
            "role_distribution_csv": OUT_DIST,
        },
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote: {OUT_ROLES}")
    print(f"[OK] wrote: {OUT_DIST}")
    print(f"[OK] wrote: {OUT_META}")
    print(f"[INFO] role distribution: {dict(dist)}")


if __name__ == "__main__":
    main()