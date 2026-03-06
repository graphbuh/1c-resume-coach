import os
import json
import re
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(".")
RESUME_DIR = PROJECT_ROOT / "data" / "resumes_txt"
RESUME_DIR.mkdir(parents=True, exist_ok=True)

RESUME_PROFILES_CSV = PROJECT_ROOT / "outputs" / "resume_profiles" / "resume_profiles.csv"
RESUME_GAP_CSV = PROJECT_ROOT / "outputs" / "resume_gap" / "resume_gap_report.csv"


st.set_page_config(
    page_title="1C Resume Coach",
    page_icon="📄",
    layout="wide",
)


def run_script(script_name: str) -> tuple[bool, str]:
    script_path = PROJECT_ROOT / "scripts" / script_name
    if not script_path.exists():
        return False, f"Файл не найден: {script_path}"

    try:
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        if result.returncode == 0:
            return True, output.strip()
        return False, output.strip()
    except Exception as e:
        return False, str(e)


def save_uploaded_resume(uploaded_file) -> Path:
    file_path = RESUME_DIR / uploaded_file.name
    content = uploaded_file.read()

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("cp1251")
        except UnicodeDecodeError:
            text = content.decode("utf-8", errors="ignore")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

    return file_path


def load_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def prettify_skill(skill: str) -> str:
    return skill.replace(":", " → ")


def parse_skills_cell(cell: str) -> list[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    return [x.strip() for x in cell.split(";") if x.strip()]


def main():
    st.title("1C Resume Coach")
    st.caption("MVP для автоматического анализа резюме и рекомендаций по росту на рынке 1С")

    with st.sidebar:
        st.header("Управление")
        st.write("1. Загрузи резюме в txt")
        st.write("2. Нажми обработку")
        st.write("3. Посмотри профиль, gap и рекомендации")

    uploaded_file = st.file_uploader(
        "Загрузи резюме в формате .txt",
        type=["txt"],
        accept_multiple_files=False,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if uploaded_file is not None:
            saved_path = save_uploaded_resume(uploaded_file)
            st.success(f"Резюме сохранено: {saved_path.name}")

    with col2:
        run_pipeline = st.button("Запустить анализ", type="primary", use_container_width=True)

    if run_pipeline:
        with st.spinner("Идёт анализ резюме..."):
            ok1, out1 = run_script("07_resume_skill_profile.py")
            ok2, out2 = run_script("08_resume_gap_to_target.py")
            ok3, out3 = run_script("09b_resume_rewrite_hints.py")

        if ok1 and ok2 and ok3:
            st.success("Анализ завершён")
        else:
            st.error("Один из шагов завершился с ошибкой")

        with st.expander("Логи выполнения"):
            st.text("07_resume_skill_profile.py\n" + out1)
            st.text("\n08_resume_gap_to_target.py\n" + out2)
            st.text("\n09b_resume_rewrite_hints.py\n" + out3)

    profile_df = load_csv_safe(RESUME_PROFILES_CSV)
    gap_df = load_csv_safe(RESUME_GAP_CSV)

    if profile_df.empty:
        st.info("Пока нет обработанных резюме. Загрузи txt и нажми «Запустить анализ».")
        return

    resume_names = profile_df["resume_file"].tolist()
    selected_resume = st.selectbox("Выбери резюме", resume_names)

    profile_row = profile_df[profile_df["resume_file"] == selected_resume].iloc[0]
    gap_row = None
    if not gap_df.empty and selected_resume in gap_df["resume_file"].values:
        gap_row = gap_df[gap_df["resume_file"] == selected_resume].iloc[0]

    st.subheader("Профиль кандидата")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Role (rules)", str(profile_row.get("rule_role", "")))
    m2.metric("Role (ML)", str(profile_row.get("ml_role", "")))
    m3.metric("Skills found", int(profile_row.get("skills_count", 0)))
    if gap_row is not None:
        m4.metric("Coverage", f"{round(float(gap_row.get('coverage', 0)) * 100, 1)}%")
    else:
        m4.metric("Coverage", "—")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Навыки", "Gap-анализ", "Советы по резюме", "Сырые данные"]
    )

    with tab1:
        st.markdown("### Найденные навыки")
        skills = parse_skills_cell(profile_row.get("skills", ""))
        if skills:
            st.write("\n".join([f"- {prettify_skill(s)}" for s in skills]))
        else:
            st.write("Навыки не найдены.")

    with tab2:
        if gap_row is None:
            st.info("Gap-отчёт пока не найден.")
        else:
            st.markdown("### Совпавшие навыки")
            matched = parse_skills_cell(gap_row.get("matched_skills", ""))
            if matched:
                st.write("\n".join([f"- {prettify_skill(s)}" for s in matched]))
            else:
                st.write("Нет совпадений.")

            st.markdown("### Недостающие навыки")
            missing = parse_skills_cell(gap_row.get("missing_skills", ""))
            if missing:
                st.write("\n".join([f"- {prettify_skill(s)}" for s in missing]))
            else:
                st.write("Недостающих навыков не найдено.")

            st.markdown("### Дополнительные навыки")
            extra = parse_skills_cell(gap_row.get("extra_skills", ""))
            if extra:
                st.write("\n".join([f"- {prettify_skill(s)}" for s in extra]))
            else:
                st.write("Дополнительных навыков нет.")

    with tab3:
        hints_path = PROJECT_ROOT / "outputs" / "resume_gap" / "resume_rewrite_hints.csv"
        hints_df = load_csv_safe(hints_path)

        if hints_df.empty or selected_resume not in hints_df["resume_file"].values:
            st.info("Советы по переписыванию резюме пока не найдены.")
        else:
            hint_row = hints_df[hints_df["resume_file"] == selected_resume].iloc[0]

            st.markdown("### Что добавить в резюме")
            add_block = str(hint_row.get("what_to_add", ""))
            st.write(add_block if add_block else "—")

            st.markdown("### Что усилить формулировками")
            improve_block = str(hint_row.get("how_to_phrase", ""))
            st.write(improve_block if improve_block else "—")

            st.markdown("### Учебные рекомендации")
            learn_block = str(hint_row.get("learning_plan", ""))
            st.write(learn_block if learn_block else "—")

    with tab4:
        st.markdown("### Строка профиля")
        st.dataframe(pd.DataFrame([profile_row]))
        if gap_row is not None:
            st.markdown("### Строка gap-отчёта")
            st.dataframe(pd.DataFrame([gap_row]))


if __name__ == "__main__":
    main()