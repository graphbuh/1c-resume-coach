import os
import re
import json
import joblib
import yaml
from typing import Dict, List, Tuple


SKILLS_YAML = os.path.join("outputs", "vocab", "skills.yaml")
ML_MODEL = os.path.join("outputs", "models", "role_mlp_pipeline.joblib")
ML_ENCODER = os.path.join("outputs", "models", "role_label_encoder.joblib")

IN_DIR = os.path.join("data", "resumes_txt")
OUT_DIR = os.path.join("outputs", "resume_profiles")
OUT_JSONL = os.path.join(OUT_DIR, "resume_profiles.jsonl")
OUT_CSV = os.path.join(OUT_DIR, "resume_profiles.csv")


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u00ad", "")
    s = s.replace("ё", "е").replace("Ё", "Е")
    s = s.lower()
    s = re.sub(r"\b1c\b", "1с", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def load_skills_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Skills YAML not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("skills.yaml must be a mapping")
    return data


def build_skill_to_aliases(skills: dict) -> Dict[Tuple[str, str], List[str]]:
    skill_to_aliases = {}
    for category, skill_map in skills.items():
        if not isinstance(skill_map, dict):
            continue
        for skill_key, spec in skill_map.items():
            if not isinstance(spec, dict):
                continue
            aliases = spec.get("aliases", [])
            norm_aliases = [normalize_text(str(a)) for a in aliases if normalize_text(str(a))]
            skill_to_aliases[(category, skill_key)] = norm_aliases
    return skill_to_aliases


def extract_skills_from_text(
    text_norm: str,
    skill_to_aliases: Dict[Tuple[str, str], List[str]]
) -> List[Dict[str, str]]:
    found = []
    for (cat, skill), aliases in skill_to_aliases.items():
        for a in aliases:
            if a and a in text_norm:
                found.append({"category": cat, "skill": skill, "matched_alias": a})
                break
    return found


def count_hits(skills: set[str], keys: list[str]) -> int:
    return sum(1 for k in keys if k in skills)


def role_scores(skills: set[str]) -> dict:
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

    programmer_score = 0.30 * count_hits(skills, programmer_keys)
    analyst_score = 0.35 * count_hits(skills, analyst_keys)
    consultant_score = 0.28 * count_hits(skills, consultant_keys)
    arch_tech_score = 0.45 * count_hits(skills, architect_technical_keys)
    arch_func_score = 0.45 * count_hits(skills, architect_functional_keys)

    if (
        "dev_language_and_metadata:language_1c" in skills
        and "queries_reports_ui:queries" in skills
    ):
        programmer_score += 0.8

    if (
        "analysis_consulting_process:requirements" in skills
        and "analysis_consulting_process:tz_spec" in skills
    ):
        analyst_score += 0.8

    if (
        "analysis_consulting_process:consulting" in skills
        and "analysis_consulting_process:user_training" in skills
    ):
        consultant_score += 0.8

    if "architecture_roles:technical_architect" in skills:
        arch_tech_score += 2.0

    if "architecture_roles:functional_architect" in skills:
        arch_func_score += 2.0

    return {
        "programmer": round(programmer_score, 4),
        "analyst": round(analyst_score, 4),
        "consultant": round(consultant_score, 4),
        "architect_technical": round(arch_tech_score, 4),
        "architect_functional": round(arch_func_score, 4),
    }


def choose_role(scores: dict) -> tuple[str, float]:
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_role, top_score = ordered[0]
    second_role, second_score = ordered[1]

    pair = {top_role, second_role}
    gap = top_score - second_score

    if top_role in {"architect_technical", "architect_functional"} and top_score >= 1.2:
        return top_role, float(top_score)

    hybrid_pairs = [
        {"analyst", "consultant"},
        {"analyst", "programmer"},
        {"consultant", "programmer"},
    ]

    if pair in hybrid_pairs and second_score >= 1.0 and gap <= 0.6:
        ordered_pair = sorted([top_role, second_role])
        return f"hybrid_{ordered_pair[0]}_{ordered_pair[1]}", float(top_score)

    if "architect_technical" in pair and scores["architect_technical"] >= 1.0:
        return "architect_technical", float(scores["architect_technical"])

    if "architect_functional" in pair and scores["architect_functional"] >= 1.0:
        return "architect_functional", float(scores["architect_functional"])

    return top_role, float(top_score)


def anonymize_text(text: str) -> str:
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL]", text)
    text = re.sub(r"@\w+", "[TG]", text)
    text = re.sub(r"(\+7|8)\s*[\(\- ]?\d{3}[\)\- ]?\s*\d{3}[\- ]?\d{2}[\- ]?\d{2}", "[PHONE]", text)
    return text


def read_resume_files(in_dir: str) -> List[dict]:
    if not os.path.exists(in_dir):
        raise FileNotFoundError(
            f"Resume dir not found: {in_dir}\n"
            f"Create it and put .txt resumes there."
        )

    rows = []
    for fname in sorted(os.listdir(in_dir)):
        if not fname.lower().endswith(".txt"):
            continue
        path = os.path.join(in_dir, fname)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        rows.append({
            "resume_file": fname,
            "text_raw": raw,
        })
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    skills = load_skills_yaml(SKILLS_YAML)
    skill_to_aliases = build_skill_to_aliases(skills)

    resumes = read_resume_files(IN_DIR)

    use_ml = os.path.exists(ML_MODEL) and os.path.exists(ML_ENCODER)
    ml_pipeline = None
    label_encoder = None

    if use_ml:
        ml_pipeline = joblib.load(ML_MODEL)
        label_encoder = joblib.load(ML_ENCODER)
        print("[INFO] loaded ML model and label encoder")
    else:
        print("[INFO] ML model not found, proceeding with rules only")

    rows_jsonl = []
    rows_csv = []

    for idx, r in enumerate(resumes):
        resume_file = r["resume_file"]
        text_raw = r["text_raw"]
        text_masked = anonymize_text(text_raw)
        text_norm = normalize_text(text_masked)

        found = extract_skills_from_text(text_norm, skill_to_aliases)
        found_skill_pairs = [f'{x["category"]}:{x["skill"]}' for x in found]
        found_skill_pairs = sorted(set(found_skill_pairs))

        scores = role_scores(set(found_skill_pairs))
        rule_role, rule_conf = choose_role(scores)

        ml_role = None
        if use_ml:
            pred_encoded = ml_pipeline.predict([text_norm])[0]
            ml_role = label_encoder.inverse_transform([pred_encoded])[0]

        rows_jsonl.append({
            "resume_id": idx,
            "resume_file": resume_file,
            "rule_role": rule_role,
            "rule_confidence": rule_conf,
            "ml_role": ml_role,
            "skills": found,
            "text_masked": text_masked,
        })

        rows_csv.append({
            "resume_id": idx,
            "resume_file": resume_file,
            "text_len": len(text_raw),
            "skills_count": len(found_skill_pairs),
            "skills": "; ".join(found_skill_pairs),
            "rule_role": rule_role,
            "rule_confidence": round(rule_conf, 4),
            "ml_role": ml_role if ml_role else "",
        })

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for row in rows_jsonl:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    import pandas as pd
    df = pd.DataFrame(rows_csv)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[OK] wrote: {OUT_JSONL}")
    print(f"[OK] wrote: {OUT_CSV}")
    print(f"[INFO] resumes processed: {len(rows_csv)}")


if __name__ == "__main__":
    main()