"""Portable publication configuration; never creates GPU allocation or credentials."""
import os
from pathlib import Path
import socket

ROOT = Path(__file__).resolve().parents[1]


def model_path(config=None):
    value = os.environ.get('ATTNCOPY_MODEL_DIR') or (config or {}).get(
        'model_path', 'data/models/Qwen3-0.6B-c1899de')
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else ROOT / path)


def require_authorized_host():
    expected = os.environ.get('ATTNCOPY_ALLOWED_HOST')
    if not expected or socket.gethostname() != expected:
        raise RuntimeError('Set ATTNCOPY_ALLOWED_HOST to an explicitly authorized experiment host')


def gpu_lock_files(source, gpu):
    base = Path(os.environ.get('ATTNCOPY_GPU_LOCK_DIR', str(source / 'outputs/bi9/.gpu-locks')))
    host = os.environ.get('ATTNCOPY_GPU_LOCK_HOST', socket.gethostname())
    if not base.is_dir():
        raise RuntimeError('Existing shared GPU lock directory is required; this code never creates it')
    return sorted(base.glob(f'{host}-gpu{gpu}*.lock'))
