from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
SUMMARY_PATH = PROJECT_ROOT / "final_analyst.txt"


def read_all_reports() -> str:
    reports = sorted(REPORT_DIR.glob("*.txt"), key=lambda path: path.name.casefold())
    if not reports:
        raise FileNotFoundError("No report files were found in outputs/reports.")

    sections = [
        f"## {report.name}\n{report.read_text(encoding='utf-8').strip()}"
        for report in reports
    ]
    return "\n\n".join(sections)


def save_final_summary(summary: str) -> Path:
    text = summary.strip()
    if not text:
        raise ValueError("The final summary cannot be empty.")
    SUMMARY_PATH.write_text(f"{text}\n", encoding="utf-8")
    return SUMMARY_PATH
