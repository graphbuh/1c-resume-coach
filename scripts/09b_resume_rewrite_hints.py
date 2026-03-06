import os
import json
import pandas as pd


IN_GAP = os.path.join("outputs", "resume_gap", "resume_gap_report.csv")

OUT_DIR = os.path.join("outputs", "resume_gap")
OUT_CSV = os.path.join(OUT_DIR, "resume_rewrite_hints.csv")
OUT_JSONL = os.path.join(OUT_DIR, "resume_rewrite_hints.jsonl")


def parse_skills_cell(cell: str) -> list[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    return [x.strip() for x in cell.split(";") if x.strip()]


def humanize_skill(skill: str) -> str:
    mapping = {
        "analysis_consulting_process:requirements": "сбор и формализация требований",
        "analysis_consulting_process:testing_uat": "тестирование и приемка доработок",
        "analysis_consulting_process:tz_spec": "подготовка технических заданий",
        "analysis_consulting_process:business_processes": "описание и моделирование бизнес-процессов",
        "analysis_consulting_process:user_training": "обучение пользователей",
        "analysis_consulting_process:consulting": "консультирование пользователей",
        "analysis_consulting_process:implementation": "участие во внедрении",
        "configs_typical:bp": "1С:Бухгалтерия предприятия",
        "configs_typical:ka": "1С:Комплексная автоматизация",
        "configs_typical:ut": "1С:Управление торговлей",
        "configs_typical:erp": "1С:ERP",
        "configs_typical:zup": "1С:ЗУП",
        "configs_typical:do": "1С:Документооборот",
        "configs_typical:upp": "1С:УПП",
        "domain_functional:warehouse": "складской учет",
        "domain_functional:regulated_accounting": "регламентированный учет",
        "domain_functional:finance_accounting": "финансовый и управленческий учет",
        "queries_reports_ui:queries": "язык запросов 1С",
        "queries_reports_ui:skd": "СКД и отчетность",
        "integrations_and_exchange:exchange_kd": "обмены и Конвертация данных",
        "integrations_and_exchange:http_rest": "интеграции через REST/HTTP",
        "sql_perf_admin:sql": "SQL",
        "core_platform:bsp": "БСП",
        "dev_language_and_metadata:language_1c": "разработка на встроенном языке 1С",
    }
    return mapping.get(skill, skill.replace(":", " -> "))


def make_what_to_add(missing_skills: list[str], role: str) -> str:
    if not missing_skills:
        return "Резюме уже хорошо покрывает рыночный профиль целевой роли."

    top = missing_skills[:6]
    human = [humanize_skill(x) for x in top]

    return (
        f"Для роли {role} стоит явно добавить в резюме опыт по следующим зонам: "
        + ", ".join(human)
        + ". Указывай не просто навык, а конкретный проект, задачу и результат."
    )


def make_how_to_phrase(missing_skills: list[str], extra_skills: list[str], role: str) -> str:
    blocks = []

    if "analysis_consulting_process:requirements" in missing_skills:
        blocks.append(
            "Добавь формулировки вроде: «Собирал и формализовывал требования пользователей, готовил постановки задач для разработчиков»."
        )

    if "analysis_consulting_process:testing_uat" in missing_skills:
        blocks.append(
            "Если участвовал в тестировании, укажи это явно: «Проводил тестирование доработок, участвовал в приемке и передаче пользователям»."
        )

    if "configs_typical:ut" in missing_skills or "configs_typical:ka" in missing_skills or "configs_typical:bp" in missing_skills:
        blocks.append(
            "Если есть опыт с типовыми конфигурациями, перечисли их отдельно в блоке «Навыки» и продублируй в описании проектов."
        )

    if "domain_functional:warehouse" in missing_skills:
        blocks.append(
            "Если есть опыт со складским учетом, закупками или логистикой, это нужно вынести в отдельную строку проекта."
        )

    if "integrations_and_exchange:exchange_kd" in extra_skills or "queries_reports_ui:skd" in extra_skills:
        blocks.append(
            "Технические навыки вроде СКД, обменов, SQL и БСП стоит не прятать в конец резюме, а подсветить в первом экране."
        )

    if not blocks:
        blocks.append(
            "Усиль формулировки через схему: задача → что сделал → какой результат получил пользователь или бизнес."
        )

    return " ".join(blocks)


def make_learning_plan(missing_skills: list[str]) -> str:
    tips = []

    if "analysis_consulting_process:requirements" in missing_skills:
        tips.append("Потренировать сбор требований и формализацию задач.")
    if "analysis_consulting_process:testing_uat" in missing_skills:
        tips.append("Добавить практику тестирования и приемки доработок.")
    if "configs_typical:bp" in missing_skills:
        tips.append("Разобрать кейсы по 1С:Бухгалтерия предприятия.")
    if "configs_typical:ka" in missing_skills:
        tips.append("Подтянуть Комплексную автоматизацию и типовые бизнес-сценарии.")
    if "configs_typical:ut" in missing_skills:
        tips.append("Освежить Управление торговлей: закупки, продажи, склад.")
    if "domain_functional:warehouse" in missing_skills:
        tips.append("Усилить знания складского контура и связанных процессов.")
    if "queries_reports_ui:queries" in missing_skills:
        tips.append("Прокачать язык запросов 1С.")
    if "queries_reports_ui:skd" in missing_skills:
        tips.append("Добавить практику по СКД и отчетности.")

    if not tips:
        return "Рекомендуется углублять проектный опыт и точнее отражать уже имеющиеся навыки в тексте резюме."

    return " ".join(tips[:6])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(IN_GAP):
        raise FileNotFoundError(f"Missing file: {IN_GAP}")

    df = pd.read_csv(IN_GAP, encoding="utf-8-sig")

    rows_csv = []
    rows_jsonl = []

    for _, row in df.iterrows():
        resume_id = int(row["resume_id"])
        resume_file = row["resume_file"]
        role = row.get("target_role", "")

        missing = parse_skills_cell(row.get("missing_skills", ""))
        extra = parse_skills_cell(row.get("extra_skills", ""))

        what_to_add = make_what_to_add(missing, role)
        how_to_phrase = make_how_to_phrase(missing, extra, role)
        learning_plan = make_learning_plan(missing)

        rows_csv.append({
            "resume_id": resume_id,
            "resume_file": resume_file,
            "target_role": role,
            "what_to_add": what_to_add,
            "how_to_phrase": how_to_phrase,
            "learning_plan": learning_plan,
        })

        rows_jsonl.append({
            "resume_id": resume_id,
            "resume_file": resume_file,
            "target_role": role,
            "what_to_add": what_to_add,
            "how_to_phrase": how_to_phrase,
            "learning_plan": learning_plan,
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