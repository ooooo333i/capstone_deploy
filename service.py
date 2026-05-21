from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse


PROJECT_ROOT = Path(__file__).resolve().parent
ENTRY_POINT = PROJECT_ROOT / "entry_point.py"
DEFAULT_SERVICE_ROOT = PROJECT_ROOT / "outputs" / "service_jobs"
DEFAULT_HAMER_ROOT = PROJECT_ROOT / "external" / "hamer"
CHUNK_SIZE = 1024 * 1024

app = FastAPI(title="CAP Motion Recovery Service")


def safe_stem(filename: str) -> str:
    stem = Path(filename).stem or "input"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return stem or "input"


def job_dir(job_id: str) -> Path:
    return DEFAULT_SERVICE_ROOT / job_id


def status_path(job_id: str) -> Path:
    return job_dir(job_id) / "status.json"


def write_status(job_id: str, payload: dict[str, Any]) -> None:
    path = status_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = time.time()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_status(job_id: str) -> dict[str, Any]:
    path = status_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="run not found")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_optional_path(raw: str | None) -> Path | None:
    if raw is None or raw.strip() == "":
        return None
    path = Path(os.path.expandvars(os.path.expanduser(raw.strip())))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def collect_artifacts(run_status: dict[str, Any]) -> dict[str, str]:
    artifact_dir = Path(run_status["artifact_dir"])
    log_path = Path(run_status["log_path"])
    candidates = {
        "merged": artifact_dir / "smplx_merged_hamer.pt",
        "report": artifact_dir / "smplx_merged_hamer.json",
        "video": artifact_dir / "smplx_merged_hamer_incam.mp4",
        "log": log_path,
    }
    return {name: str(path) for name, path in candidates.items() if path.exists()}


def run_pipeline(job_id: str, command: list[str]) -> None:
    run_status = read_status(job_id)
    log_path = Path(run_status["log_path"])
    started_at = time.time()
    run_status.update({"state": "running", "started_at": started_at, "command": command})
    write_status(job_id, run_status)

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
    except Exception as exc:
        run_status.update(
            {
                "state": "failed",
                "error": str(exc),
                "finished_at": time.time(),
                "elapsed_sec": time.time() - started_at,
            }
        )
        write_status(job_id, run_status)
        return

    run_status.update(
        {
            "state": "succeeded" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "finished_at": time.time(),
            "elapsed_sec": time.time() - started_at,
        }
    )
    run_status["artifacts"] = collect_artifacts(run_status)
    write_status(job_id, run_status)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "project_root": str(PROJECT_ROOT),
        "entry_point": ENTRY_POINT.exists(),
    }


@app.get("/runs")
def list_runs() -> list[dict[str, Any]]:
    if not DEFAULT_SERVICE_ROOT.exists():
        return []
    runs = []
    for path in sorted(DEFAULT_SERVICE_ROOT.glob("*/status.json"), reverse=True):
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    return runs


@app.post("/runs")
async def create_run(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    static_cam: bool = Form(False),
    use_dpvo: bool = Form(False),
    f_mm: int | None = Form(None),
    force: bool = Form(False),
    render_preview: bool = Form(False),
    skip_result_video: bool = Form(True),
    person_track_id: int | None = Form(None),
    hamer_checkpoint: str | None = Form(None),
    hamer_batch_size: int = Form(1),
    hamer_rescale_factor: float = Form(2.5),
    hand_min_conf: float = Form(0.35),
) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    root = job_dir(job_id)
    input_dir = root / "input"
    output_root = root / "output"
    input_dir.mkdir(parents=True, exist_ok=True)

    stem = safe_stem(video.filename or "input")
    suffix = Path(video.filename or "").suffix.lower() or ".mp4"
    input_path = input_dir / f"{stem}{suffix}"
    with input_path.open("wb") as handle:
        while True:
            chunk = await video.read(CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)

    if input_path.stat().st_size == 0:
        raise HTTPException(status_code=400, detail="empty video upload")

    hamer_root = resolve_optional_path(os.environ.get("CAP_HAMER_ROOT")) or DEFAULT_HAMER_ROOT
    checkpoint = resolve_optional_path(hamer_checkpoint) or resolve_optional_path(os.environ.get("CAP_HAMER_CHECKPOINT"))

    command = [
        sys.executable,
        str(ENTRY_POINT),
        "--video",
        str(input_path),
        "--output-root",
        str(output_root),
        "--hamer-root",
        str(hamer_root),
        "--auto-person",
        "--no-interactive",
        "--person-select-ui",
        "auto",
        "--hamer-batch-size",
        str(hamer_batch_size),
        "--hamer-rescale-factor",
        str(hamer_rescale_factor),
        "--hand-min-conf",
        str(hand_min_conf),
    ]
    if static_cam:
        command.append("--static-cam")
    if use_dpvo:
        command.append("--use-dpvo")
    if f_mm is not None:
        command.extend(["--f-mm", str(f_mm)])
    if force:
        command.append("--force")
    if render_preview:
        command.append("--render-preview")
    if skip_result_video:
        command.append("--skip-result-video")
    if person_track_id is not None:
        command.extend(["--person-track-id", str(person_track_id)])
    if checkpoint is not None:
        command.extend(["--hamer-checkpoint", str(checkpoint)])

    initial_status = {
        "id": job_id,
        "state": "queued",
        "created_at": time.time(),
        "input_video": str(input_path),
        "output_root": str(output_root),
        "artifact_dir": str(output_root / input_path.stem),
        "log_path": str(root / "pipeline.log"),
        "artifacts": {},
    }
    write_status(job_id, initial_status)
    background_tasks.add_task(run_pipeline, job_id, command)
    return read_status(job_id)


@app.get("/runs/{job_id}")
def get_run(job_id: str) -> dict[str, Any]:
    run_status = read_status(job_id)
    run_status["artifacts"] = collect_artifacts(run_status)
    return run_status


@app.get("/runs/{job_id}/artifacts/{name}")
def get_artifact(job_id: str, name: str) -> FileResponse:
    run_status = read_status(job_id)
    artifacts = collect_artifacts(run_status)
    if name not in artifacts:
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(artifacts[name])
