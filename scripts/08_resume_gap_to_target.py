import os
import json
from collections import Counter, defaultdict

import pandas as pd


IN_RESUME_PROFILES = os.path.join("outputs", "resume_profiles", "resume_profiles.csv")
IN_VACANCY_SKILLS = os.path.join("outputs", "skills", "vacancy_skills.csv")
IN_VACANCY_ROLES = os.path.join("outputs", "roles", "vacancy_roles.csv")

OUT_DIR = os.path.join("outputs", "resume_gap")
OUT_CSV = os.path.join(OUT_DIR, "resume_gap_report.csv")
OUT_JSONL = os.path.join(OUT_DIR, "resume_gap_report.jsonl")


def parse_skills_cell(cell: str) -> set[str]:
    if not isinstance(cell, str) or not cell.strip():
        return set()
    parts = [p.strip() for p in cell.split(";")]
    return {p for p in parts if p}


def build_role_skill_profiles(vacancy_skills_df: pd.DataFrame, vacancy_roles_df: pd.DataFrame) -> dict:
    """
    Строим профиль навыков по каждой роли:
    role -> Counter(skill -> count)
    """
    merged = vacancy_skills_df.merge(
        vacancy_roles_df[["vacancy_id", "role"]],
        on="vacancy_id",
        how="inner"
    )

    role_skill_counter = defaultdict(Counter)
    role_vacancy_count = Counter()

    for _, row in merged.iterrows():
        role = row["role"]
        skills = parse_skills_cell(row.get("skills", ""))
        role_vacancy_count[role] += 1

        for skill in skills:
            role_skill_counter[role][skill] += 1

    return {
        "role_skill_counter": role_skill_counter,
        "role_vacancy_count": role_vacancy_count
    }


def top_role_skills(role: str, role_profiles: dict, top_n: int = 20, min_share: float = 0.15) -> list[dict]:
    """
    Возвращает top skills по роли с частотами и долями.
    min_share = минимальная доля вакансий роли, где встречается навык
    """
    counter = role_profiles["role_skill_counter"].get(role, Counter())
    total = role_profiles["role_vacancy_count"].get(role, 0)

    if total == 0:
        return []

    rows = []
    for skill, count in counter.items():
        share = count / total
        if share >= min_share:
            rows.append({
                "skill": skill,
                "count": count,
                "share": round(share, 4)
            })

    rows = sorted(rows, key=lambda x: (-x["share"], -x["count"], x["skill"]))
    return rows[:top_n]


def suggest_learning_actions(missing_skills: list[str]) -> list[str]:
    """
    Простые рекомендации по развитию на основе недостающих навыков.
    """
    tips = []

    mapping = {
        "queries_reports_ui:queries": "Прокачать язык запросов 1С и типовые конструкции выборок.",
        "queries_reports_ui:skd": "Добавить опыт работы со СКД и сложными отчетами.",
        "queries_reports_ui:query_optimization": "Изучить оптимизацию запросов, временные таблицы и планы запросов.",
        "dev_language_and_metadata:language_1c": "Усилить практику разработки на встроенном языке 1С.",
        "core_platform:bsp": "Освоить БСП и типовые механизмы повторного использования.",
        "integrations_and_exchange:http_rest": "Добавить интеграции через REST/HTTP-сервисы.",
        "integrations_and_exchange:exchange_kd": "Подтянуть обмены и Конвертацию данных (КД2/КД3).",
        "integrations_and_exchange:message_brokers": "Изучить брокеры сообщений и событийные интеграции (Kafka/RabbitMQ).",
        "sql_perf_admin:sql": "Укрепить SQL и работу с источниками данных.",
        "sql_perf_admin:optimization": "Добавить кейсы по производительности и оптимизации.",
        "analysis_consulting_process:requirements": "Потренировать сбор и формализацию требований.",
        "analysis_consulting_process:tz_spec": "Научиться писать ТЗ и структурировать задачи для разработки.",
        "analysis_consulting_process:business_processes": "Усилить навыки моделирования и описания бизнес-процессов.",
        "analysis_consulting_process:user_training": "Добавить опыт обучения пользователей и сопровождения внедрений.",
        "analysis_consulting_process:consulting": "Развивать навыки консультаций и коммуникации с пользователями.",
        "analysis_consulting_process:implementation": "Наработать опыт внедрения и сопровождения систем.",
        "architecture_functional_signals:process_modeling": "Изучить BPMN/UML/IDEF и практику моделирования процессов.",
        "architecture_functional_signals:functional_architecture": "Добавить опыт проектирования функциональной архитектуры.",
        "architecture_technical_signals:technical_architecture": "Прокачать техническую архитектуру решений на 1С.",
        "architecture_technical_signals:integration_architecture": "Разобраться в интеграционной архитектуре и потоках данных.",
        "leadership_general:tech_lead": "Развивать лидерские навыки: декомпозиция, ревью, наставничество.",
    }

    for skill in missing_skills:
        if skill in mapping:
            tips.append(mapping[skill])

    # убрать повторы, сохранить порядок
    uniq = []
    seen = set()
    for t in tips:
        if t not in seen:
            uniq.append(t)
            seen.add(t)

    return uniq[:8]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(IN_RESUME_PROFILES):
        raise FileNotFoundError(f"Missing file: {IN_RESUME_PROFILES}")
    if not os.path.exists(IN_VACANCY_SKILLS):
        raise FileNotFoundError(f"Missing file: {IN_VACANCY_SKILLS}")
    if not os.path.exists(IN_VACANCY_ROLES):
        raise FileNotFoundError(f"Missing file: {IN_VACANCY_ROLES}")

    resume_df = pd.read_csv(IN_RESUME_PROFILES, encoding="utf-8-sig")
    vacancy_skills_df = pd.read_csv(IN_VACANCY_SKILLS, encoding="utf-8-sig")
    vacancy_roles_df = pd.read_csv(IN_VACANCY_ROLES, encoding="utf-8-sig")

    role_profiles = build_role_skill_profiles(vacancy_skills_df, vacancy_roles_df)

    rows_csv = []
    rows_jsonl = []

    for _, row in resume_df.iterrows():
        resume_id = int(row["resume_id"])
        resume_file = row["resume_file"]
        resume_skills = parse_skills_cell(row.get("skills", ""))

        detected_role = row.get("rule_role", "")
        ml_role = row.get("ml_role", "")

        # целевая роль: если ML есть и не пустой, можно хранить отдельно, но основная = rule_role
        target_role = detected_role

        role_top_skills = top_role_skills(target_role, role_profiles, top_n=20, min_share=0.15)
        market_skills = [x["skill"] for x in role_top_skills]

        matched = sorted([s for s in market_skills if s in resume_skills])
        missing = sorted([s for s in market_skills if s not in resume_skills])
        extra = sorted([s for s in resume_skills if s not in set(market_skills)])

        coverage = 0.0
        if len(market_skills) > 0:
            coverage = len(matched) / len(market_skills)

        recommendations = []
        for s in missing[:10]:
            recommendations.append(f"Добавить/подтвердить навык в резюме: {s}")

        learning_tips = suggest_learning_actions(missing)

        rows_csv.append({
            "resume_id": resume_id,
            "resume_file": resume_file,
            "detected_role": detected_role,
            "ml_role": ml_role,
            "target_role": target_role,
            "resume_skills_count": len(resume_skills),
            "market_skills_count": len(market_skills),
            "matched_count": len(matched),
            "missing_count": len(missing),
            "coverage": round(coverage, 4),
            "matched_skills": "; ".join(matched),
            "missing_skills": "; ".join(missing),
            "extra_skills": "; ".join(extra),
            "recommendations": " | ".join(recommendations[:8]),
            "learning_tips": " | ".join(learning_tips[:8]),
        })

        rows_jsonl.append({
            "resume_id": resume_id,
            "resume_file": resume_file,
            "detected_role": detected_role,
            "ml_role": ml_role,
            "target_role": target_role,
            "coverage": round(coverage, 4),
            "market_top_skills": role_top_skills,
            "matched_skills": matched,
            "missing_skills": missing,
            "extra_skills": extra,
            "recommendations": recommendations[:8],
            "learning_tips": learning_tips[:8],
        })

    out_df = pd.DataFrame(rows_csv)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in rows_jsonl:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] wrote: {OUT_CSV}")
    print(f"[OK] wrote: {OUT_JSONL}")
    print(f"[INFO] processed resumes: {len(rows_csv)}")


if __name__ == "__main__":
    main()