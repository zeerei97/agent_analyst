from __future__ import annotations

import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CHART_DIR = PROJECT_ROOT / "outputs" / "charts"
PENDING_CHART_DIR = PROJECT_ROOT / "outputs" / ".pending_charts"

ChartType = Literal["line", "bar", "histogram", "box", "scatter", "heatmap"]


class ChartValidationError(ValueError):
    """Raised when a chart request is incompatible with the selected columns."""


def save_dataset_upload(filename: str, content: bytes) -> Path:
    if Path(filename).suffix.lower() != ".csv":
        raise ValueError("Dataset upload must be a .csv file.")
    if not content:
        raise ValueError("The uploaded CSV file is empty.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).stem).strip("._")
    if not safe_stem:
        safe_stem = "dataset"
    destination = DATA_DIR / f"{safe_stem}_{uuid.uuid4().hex[:8]}.csv"
    destination.write_bytes(content)
    return destination


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path).expanduser().resolve()
    data_root = DATA_DIR.resolve()
    if path != data_root and data_root not in path.parents:
        raise ValueError("Dataset paths must point to a CSV inside the data/ directory.")
    if path.suffix.lower() != ".csv":
        raise ValueError("Only .csv datasets are supported.")
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path.name}")

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("The selected CSV has no data rows.")
    if frame.columns.empty:
        raise ValueError("The selected CSV has no columns.")
    return frame


def get_column_metadata(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column): str(dtype) for column, dtype in frame.dtypes.items()}


def _is_numeric(series: pd.Series) -> bool:
    return is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype)


def _validate_chart_request(
    frame: pd.DataFrame, chart_type: ChartType, columns: list[str]
) -> None:
    if any(column not in frame.columns for column in columns):
        unknown = [column for column in columns if column not in frame.columns]
        raise ChartValidationError(f"Unknown column(s): {', '.join(unknown)}.")

    expected_count = {
        "line": 2,
        "bar": 2,
        "histogram": 1,
        "box": 1,
        "scatter": 2,
    }
    if chart_type in expected_count and len(columns) != expected_count[chart_type]:
        raise ChartValidationError(
            f"{chart_type.title()} charts require {expected_count[chart_type]} "
            f"column(s); received {len(columns)}."
        )
    if chart_type == "heatmap" and columns:
        raise ChartValidationError("Heatmaps use all numeric columns; do not select columns.")

    if chart_type in {"line", "bar"} and not _is_numeric(frame[columns[1]]):
        raise ChartValidationError(
            f"{chart_type.title()} charts require a numeric second column for the value axis."
        )
    if chart_type == "histogram" and not _is_numeric(frame[columns[0]]):
        raise ChartValidationError("Histograms require one numeric column.")
    if chart_type == "box" and not _is_numeric(frame[columns[0]]):
        raise ChartValidationError("Box plots require one numeric column.")
    if chart_type == "scatter" and not all(_is_numeric(frame[column]) for column in columns):
        raise ChartValidationError("Scatter plots require two numeric columns.")
    if chart_type == "heatmap":
        numeric_columns = [column for column in frame if _is_numeric(frame[column])]
        if len(numeric_columns) < 2:
            raise ChartValidationError("A correlation heatmap requires at least two numeric columns.")


def create_chart(
    frame: pd.DataFrame,
    chart_type: ChartType,
    columns: list[str],
    title: str,
) -> Path:
    _validate_chart_request(frame, chart_type, columns)
    PENDING_CHART_DIR.mkdir(parents=True, exist_ok=True)
    safe_title = re.sub(r"[^A-Za-z0-9_-]+", "_", title).strip("_") or chart_type
    filename = f"{safe_title[:48]}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.png"
    destination = PENDING_CHART_DIR / filename

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    try:
        if chart_type == "line":
            axis.plot(frame[columns[0]], frame[columns[1]], marker="o")
            axis.set_xlabel(columns[0])
            axis.set_ylabel(columns[1])
        elif chart_type == "bar":
            grouped = frame.groupby(columns[0], dropna=False)[columns[1]].mean().sort_values(
                ascending=False
            )
            axis.bar(grouped.index.astype(str), grouped.values)
            axis.set_xlabel(columns[0])
            axis.set_ylabel(f"Mean {columns[1]}")
            axis.tick_params(axis="x", labelrotation=35)
        elif chart_type == "histogram":
            axis.hist(frame[columns[0]].dropna(), bins="auto", edgecolor="white")
            axis.set_xlabel(columns[0])
            axis.set_ylabel("Count")
        elif chart_type == "box":
            axis.boxplot(frame[columns[0]].dropna(), orientation="vertical")
            axis.set_ylabel(columns[0])
        elif chart_type == "scatter":
            axis.scatter(frame[columns[0]], frame[columns[1]], alpha=0.7)
            axis.set_xlabel(columns[0])
            axis.set_ylabel(columns[1])
        else:
            numeric = frame.select_dtypes(include="number")
            correlation = numeric.corr()
            image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
            axis.set_xticks(
                range(len(correlation.columns)), correlation.columns, rotation=35, ha="right"
            )
            axis.set_yticks(range(len(correlation.columns)), correlation.columns)
            figure.colorbar(image, ax=axis, label="Correlation")

        axis.set_title(title)
        figure.savefig(destination, dpi=150)
    finally:
        plt.close(figure)
    return destination


def approve_chart(preview_path: str | Path) -> Path:
    source = Path(preview_path).resolve()
    pending_root = PENDING_CHART_DIR.resolve()
    if pending_root not in source.parents or not source.is_file():
        raise FileNotFoundError("The pending chart preview is no longer available.")
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    destination = CHART_DIR / source.name
    shutil.move(str(source), destination)
    return destination


def discard_chart(preview_path: str | Path) -> None:
    source = Path(preview_path).resolve()
    pending_root = PENDING_CHART_DIR.resolve()
    if pending_root in source.parents and source.is_file():
        source.unlink()