"""Append reproducible IQA result rows locally and, optionally, to Google Sheets."""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

RESULT_COLUMNS = (
    "timestamp_utc",
    "run_id",
    "source_run_id",
    "experiment",
    "run_name",
    "evaluation",
    "dataset",
    "backbone",
    "method",
    "seed",
    "epochs",
    "latency_p50_ms",
    "latency_p95_ms",
    "peak_memory_mb",
    "images",
    "srcc",
    "plcc",
    "srcc_per_reference",
    "images_per_second",
    "milliseconds_per_image",
    "head_size_mb",
    "model_parameter_size_mb",
    "config_path",
)

DEFAULT_SERVICE_ACCOUNT_FILE = Path.home() / ".config/conditioned-iqa/google-service-account.json"


def default_service_account_file() -> str | None:
    """Find the local service-account key without putting its path in experiment configs."""
    configured = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if configured:
        return configured
    if DEFAULT_SERVICE_ACCOUNT_FILE.is_file():
        return str(DEFAULT_SERVICE_ACCOUNT_FILE)
    return None


def add_reporting_arguments(parser) -> None:
    """Add common local/Google Sheets reporting options to an argument parser."""
    parser.add_argument("--results-csv", default="runs/results.csv",
                        help="append every result row to this local CSV")
    parser.add_argument("--google-sheet-id", default=os.getenv("IQA_GOOGLE_SHEET_ID"),
                        help="Google spreadsheet ID; also reads IQA_GOOGLE_SHEET_ID")
    parser.add_argument("--google-worksheet", default=os.getenv("IQA_GOOGLE_WORKSHEET", "Results"),
                        help="worksheet name inside the shared spreadsheet")
    parser.add_argument("--google-service-account-file", default=default_service_account_file(),
                        help="service-account JSON path; also reads GOOGLE_APPLICATION_CREDENTIALS")
    parser.add_argument("--google-service-account-json", default=os.getenv("IQA_GOOGLE_SERVICE_ACCOUNT_JSON"),
                        help="service-account JSON (prefer the file or environment variable, never commit this)")


def size_megabytes(module) -> float:
    """Return parameter storage size, suitable for a deployment-size column."""
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters()) / 1024**2


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile for latency samples."""
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction / 100 * len(ordered)) - 1))
    return ordered[index]


def measure_latency_memory(
    forward: Callable[[], object],
    device: object,
    *,
    warmup: int = 10,
    repeats: int = 50,
) -> tuple[float, float, float]:
    """Measure inference latency percentiles and GPU peak memory.

    ``forward`` must run one complete metric inference step (image -> score).
    Latency is measured with CUDA events when available; peak memory is the
    CUDA allocator peak observed after warmup and timed runs.
    """
    import torch

    use_cuda = getattr(device, "type", "") == "cuda" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(warmup):
        forward()
    if use_cuda:
        torch.cuda.synchronize(device)

    samples: list[float] = []
    for _ in range(repeats):
        if use_cuda:
            # ``device`` need not be the process's current CUDA device (the
            # experiment runner intentionally places jobs on different GPUs).
            # Recording on the current stream of the requested device avoids
            # cross-device events that can remain incomplete at elapsed_time.
            stream = torch.cuda.current_stream(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            forward()
            end.record(stream)
            stream.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            start = time.perf_counter()
            forward()
            samples.append((time.perf_counter() - start) * 1000)

    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2 if use_cuda else math.nan
    )
    return (
        round(percentile(samples, 50), 3),
        round(percentile(samples, 95), 3),
        round(peak_memory_mb, 1),
    )


def measure_flops(forward: Callable[[], object]) -> float:
    """Return total FLOPs for one forward pass."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    with torch.no_grad():
        with FlopCounterMode(display=False) as counter:
            forward()
    return float(counter.get_total_flops())


@dataclass
class ResultReporter:
    results_csv: Path
    google_sheet_id: str | None = None
    google_worksheet: str = "Results"
    service_account_file: str | None = None
    service_account_json: str | None = None

    @classmethod
    def from_args(cls, args) -> ResultReporter:
        return cls(
            results_csv=Path(args.results_csv),
            google_sheet_id=args.google_sheet_id,
            google_worksheet=args.google_worksheet,
            service_account_file=args.google_service_account_file,
            service_account_json=args.google_service_account_json,
        )

    def append(self, rows: list[dict[str, Any]]) -> None:
        """Write locally first, then append the same schema to Google when configured."""
        normalized = [self._normalize(row) for row in rows]
        self._append_csv(normalized)
        if self.google_sheet_id:
            self._append_google(normalized)

    @staticmethod
    def _normalize(row: dict[str, Any]) -> dict[str, Any]:
        result = {key: "" for key in RESULT_COLUMNS}
        result.update(row)
        result["timestamp_utc"] = result["timestamp_utc"] or datetime.now(UTC).isoformat()
        return result

    def _append_csv(self, rows: list[dict[str, Any]]) -> None:
        import fcntl

        self.results_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.results_csv.open("a+", newline="", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                stream.seek(0)
                existing: list[dict[str, Any]] = [
                    self._normalize(row) for row in csv.DictReader(stream)
                ]
                stream.seek(0)
                stream.truncate()
                writer = csv.DictWriter(
                    stream, fieldnames=RESULT_COLUMNS, extrasaction="ignore", lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(existing + rows)
                stream.flush()
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _append_google(self, rows: list[dict[str, Any]]) -> None:
        try:
            import gspread
        except ImportError as error:
            raise RuntimeError(
                "Google Sheets reporting requires `uv sync --extra sheets` or installing gspread."
            ) from error
        if self.service_account_json:
            client = gspread.service_account_from_dict(json.loads(self.service_account_json))
        elif self.service_account_file:
            client = gspread.service_account(filename=self.service_account_file)
        else:
            raise RuntimeError(
                "A Google Sheet ID was set but no service-account credential was supplied. "
                "Set GOOGLE_APPLICATION_CREDENTIALS or IQA_GOOGLE_SERVICE_ACCOUNT_JSON."
            )
        spreadsheet = client.open_by_key(self.google_sheet_id)
        try:
            worksheet = spreadsheet.worksheet(self.google_worksheet)
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=self.google_worksheet, rows=1000, cols=len(RESULT_COLUMNS))
        header = worksheet.row_values(1)
        if header != list(RESULT_COLUMNS):
            worksheet.update("A1", [list(RESULT_COLUMNS)], value_input_option="RAW")
        worksheet.append_rows(
            [[row[column] for column in RESULT_COLUMNS] for row in rows], value_input_option="USER_ENTERED"
        )
