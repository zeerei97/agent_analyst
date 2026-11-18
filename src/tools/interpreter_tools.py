from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHART_DIR = PROJECT_ROOT / "outputs" / "charts"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"


def get_chart_path(filename: str) -> Path:
    requested = Path(filename.strip())
    if requested.name != filename.strip() or requested.suffix.lower() not in {
        ".png",
        ".jpg",
        ".jpeg",
    }:
        raise ValueError("Provide the exact chart filename ending in .png, .jpg, or .jpeg.")

    path = (CHART_DIR / requested.name).resolve()
    if CHART_DIR.resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError(f"Chart not found in outputs/charts: {requested.name}")
    return path


def save_report(chart_name: str, content: str) -> Path:
    text = content.strip()
    if not text:
        raise ValueError("An empty interpretation cannot be saved.")
    stem = Path(chart_name).stem
    if not stem or Path(chart_name).name != chart_name:
        raise ValueError("The chart name is invalid.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = (REPORT_DIR / f"{stem}.txt").resolve()
    if REPORT_DIR.resolve() not in report_path.parents:
        raise ValueError("The report filename must stay inside outputs/reports.")
    report_path.write_text(f"{text}\n", encoding="utf-8")
    return report_path