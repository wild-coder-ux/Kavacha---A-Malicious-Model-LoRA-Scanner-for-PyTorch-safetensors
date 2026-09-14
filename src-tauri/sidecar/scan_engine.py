"""AI Kavacha - Sidecar Engine.

A defensive scanner for ML model/adaptor files.

Design goals:
- Never deserialize the target model inside this process.
- Run pickle-focused scanners in isolated subprocesses/processes.
- Hash the file before returning a result.
- Inspect LoRA safetensors numerically without loading the full model into memory.
- Fail closed: scanner errors become explicit "error" checks rather than "clean".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import queue
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open
from scipy.linalg import svdvals


PICKLE_EXTS = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}
SAFETENSORS_EXTS = {".safetensors"}

DEFAULT_TOOL_TIMEOUT = 30
DEFAULT_HASH_CHUNK = 1024 * 1024
DEFAULT_MAX_MATRIX_DIM = 4096
DEFAULT_SPECTRAL_THRESHOLD = 0.85
MAX_DETAIL_LENGTH = 500


def result(label: str, status: str, detail: str) -> dict[str, str]:
    """Build a consistent check result."""
    return {
        "label": label,
        "status": status,
        "detail": detail[:MAX_DETAIL_LENGTH],
    }


def sha256(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(DEFAULT_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _tool_exists(command: str) -> bool:
    """Return True when a command is available on PATH."""
    return shutil.which(command) is not None


def _pscan_worker(file_path: str, conn: Any) -> None:
    """Run picklescan in an isolated process and return a serializable result."""
    try:
        from picklescan.scanner import scan_file_path, SafetyLevel

        scan_result = scan_file_path(file_path)

        flagged = [
            g for g in getattr(scan_result, "globals", [])
            if getattr(g, "safety", None) in (SafetyLevel.Dangerous, SafetyLevel.Suspicious)
        ]

        if not flagged:
            payload = result(
                "picklescan",
                "clean",
                "No malicious opcodes reported",
            )
        else:
            details = [f"{g.safety.value}: {g.module}.{g.name}" for g in flagged]
            severity = "malicious" if any(
                g.safety == SafetyLevel.Dangerous for g in flagged
            ) else "suspicious"

            payload = result(
                "picklescan",
                severity,
                "; ".join(details),
            )

        conn.send(payload)
    except Exception as exc:  # scanner is third-party code
        try:
            conn.send(result("picklescan", "error", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def run_picklescan(file_path: str, timeout: int = DEFAULT_TOOL_TIMEOUT) -> dict[str, str]:
    """Run picklescan with a hard timeout.

    A multiprocessing context is used so a stuck scanner cannot block the
    main engine indefinitely.
    """
    if not _tool_exists(sys.executable):
        return result("picklescan", "error", "Python executable unavailable")

    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(
        target=_pscan_worker,
        args=(file_path, child_conn),
        daemon=True,
    )

    try:
        process.start()
        child_conn.close()
        process.join(timeout)

        if process.is_alive():
            process.terminate()
            process.join(2)
            return result("picklescan", "error", f"Timeout after {timeout}s")

        if parent_conn.poll(0.2):
            payload = parent_conn.recv()
            if isinstance(payload, dict):
                return payload

        return result(
            "picklescan",
            "error",
            f"Scanner exited without a result (exit={process.exitcode})",
        )
    except Exception as exc:
        if process.is_alive():
            process.terminate()
            process.join(2)
        return result("picklescan", "error", f"{type(exc).__name__}: {exc}")
    finally:
        try:
            parent_conn.close()
        except Exception:
            pass


def _run_subprocess(
    label: str,
    command: list[str],
    timeout: int = DEFAULT_TOOL_TIMEOUT,
) -> tuple[int, str, str] | None:
    """Run a tool without invoking a shell."""
    if not shutil.which(command[0]):
        return None

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return completed.returncode, completed.stdout or "", completed.stderr or ""
    except subprocess.TimeoutExpired:
        return (-999, "", f"Timeout after {timeout}s")
    except OSError as exc:
        return (-998, "", f"{type(exc).__name__}: {exc}")


def run_fickling(file_path: str, timeout: int = DEFAULT_TOOL_TIMEOUT) -> dict[str, str]:
    """Run Fickling and distinguish a tool error from a malicious finding."""
    completed = _run_subprocess("fickling", ["fickling", file_path], timeout)
    if completed is None:
        return result("fickling", "error", "fickling is not installed or not on PATH")

    return_code, stdout, stderr = completed
    output = f"{stderr.strip()} {stdout.strip()}".strip()

    if return_code == -999:
        return result("fickling", "error", output)
    if return_code == -998:
        return result("fickling", "error", output)

    lower = output.lower()

    # These messages indicate that the input was not a pickle that Fickling
    # could analyze; they are not evidence that the file is clean.
    if "no pickle" in lower or "not a pickle" in lower:
        return result("fickling", "skipped", "Input is not a supported pickle")

    # A successful Fickling run is not automatically proof that a model is
    # safe. The scanner's own finding text is authoritative when present.
    suspicious_markers = (
        "malicious",
        "dangerous",
        "suspicious",
        "arbitrary code",
        "os.system",
        "subprocess",
        "exec(",
    )
    if any(marker in lower for marker in suspicious_markers):
        return result("fickling", "malicious", output or "Suspicious pickle content")

    if return_code != 0:
        return result(
            "fickling",
            "error",
            output or f"Fickling exited with code {return_code}",
        )

    return result("fickling", "clean", output[:MAX_DETAIL_LENGTH] or "No findings reported")


def run_modelscan(file_path: str, timeout: int = DEFAULT_TOOL_TIMEOUT) -> dict[str, str]:
    """Run ModelScan.

    ModelScan CLI output can vary across versions, so a non-zero exit is
    treated as an error unless the output clearly reports a security finding.
    """
    completed = _run_subprocess(
        "modelscan",
        ["modelscan", "-p", file_path],
        timeout,
    )
    if completed is None:
        return result("modelscan", "error", "modelscan is not installed or not on PATH")

    return_code, stdout, stderr = completed
    output = f"{stderr.strip()} {stdout.strip()}".strip()
    lower = output.lower()

    if return_code == -999 or return_code == -998:
        return result("modelscan", "error", output)

    malicious_markers = (
        "malicious",
        "critical",
        "high severity",
        "suspicious",
        "unsafe",
    )
    if any(marker in lower for marker in malicious_markers):
        return result("modelscan", "malicious", output or "Security finding reported")

    if return_code != 0:
        return result(
            "modelscan",
            "error",
            output or f"ModelScan exited with code {return_code}",
        )

    return result("modelscan", "clean", output[:MAX_DETAIL_LENGTH] or "No issues reported")


def _read_safetensors_index(file_path: str, max_header_bytes: int = 64 * 1024 * 1024) -> dict[str, dict[str, Any]]:
    """Read the safetensors header without invoking a tensor backend.

    This fallback is important on installations where safetensors/NumPy cannot
    represent BF16 tensors. Only metadata is parsed here; no model code is run.
    """
    import struct

    file_size = os.path.getsize(file_path)
    if file_size < 8:
        raise ValueError("File is too small to be a safetensors file")

    with open(file_path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        if header_len <= 0 or header_len > max_header_bytes:
            raise ValueError(f"Invalid safetensors header length: {header_len}")
        header_bytes = fh.read(header_len)

    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid safetensors header JSON: {exc}") from exc

    data_start = 8 + header_len
    tensors: dict[str, dict[str, Any]] = {}

    for key, meta in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(meta, dict):
            raise ValueError(f"Invalid tensor metadata for {key}")

        dtype = meta.get("dtype")
        shape = meta.get("shape")
        offsets = meta.get("data_offsets")

        if dtype not in {"F16", "F32", "F64", "BF16"}:
            # Spectral scanning only needs floating point tensors.
            continue
        if not isinstance(shape, list) or not all(isinstance(x, int) and x >= 0 for x in shape):
            raise ValueError(f"Invalid shape for tensor {key}")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"Invalid data offsets for tensor {key}")

        begin, end = offsets
        if not isinstance(begin, int) or not isinstance(end, int) or begin < 0 or end < begin:
            raise ValueError(f"Invalid data offsets for tensor {key}")

        absolute_end = data_start + end
        if absolute_end > file_size:
            raise ValueError(f"Tensor {key} points beyond end of file")

        tensors[key] = {
            "dtype": dtype,
            "shape": tuple(shape),
            "offset": data_start + begin,
            "nbytes": end - begin,
        }

    return tensors


def _raw_safetensors_tensor(file_path: str, key: str, index: dict[str, dict[str, Any]]) -> np.ndarray:
    """Read one floating-point safetensors tensor, including BF16."""
    import mmap

    meta = index[key]
    shape = meta["shape"]
    dtype = meta["dtype"]
    offset = meta["offset"]
    nbytes = meta["nbytes"]

    dtype_map = {
        "F16": np.float16,
        "F32": np.float32,
        "F64": np.float64,
    }

    expected_items = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if dtype == "BF16":
        expected_bytes = expected_items * 2
    else:
        np_dtype = dtype_map[dtype]
        expected_bytes = expected_items * np.dtype(np_dtype).itemsize

    if expected_bytes != nbytes:
        raise ValueError(
            f"Tensor {key}: size mismatch (header={nbytes}, expected={expected_bytes})"
        )

    if nbytes == 0:
        return np.empty(shape, dtype=np.float32)

    with open(file_path, "rb") as fh:
        # mmap offsets must be aligned to the platform allocation granularity.
        granularity = mmap.ALLOCATIONGRANULARITY
        map_offset = (offset // granularity) * granularity
        delta = offset - map_offset
        mm = mmap.mmap(
            fh.fileno(),
            length=delta + nbytes,
            access=mmap.ACCESS_READ,
            offset=map_offset,
        )
        try:
            dtype_for_read = np.uint16 if dtype == "BF16" else np_dtype
            raw = np.frombuffer(mm, dtype=dtype_for_read, count=expected_items, offset=delta).copy()
        finally:
            mm.close()

    raw = raw.reshape(shape)

    if dtype == "BF16":
        # BF16 stores the upper 16 bits of an IEEE-754 float32.
        return (raw.astype(np.uint32) << 16).view(np.float32)
    return raw.astype(np.float32, copy=False)


def _numpy_tensor(
    file_path: str,
    key: str,
    index: dict[str, dict[str, Any]],
) -> np.ndarray:
    """Read one safetensors tensor as float32.

    Torch is attempted first because it natively understands BF16. If that
    backend is unavailable or fails, use the safetensors header directly so
    BF16 still works without requiring torch.
    """
    try:
        import torch

        with safe_open(file_path, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(key)
            return tensor.float().cpu().numpy()
    except Exception:
        if key not in index:
            raise KeyError(f"Tensor not present in safetensors index: {key}")
        return _raw_safetensors_tensor(file_path, key, index)


def pair_key(key: str) -> str:
    """Convert a LoRA down/A key to its paired up/B key."""
    replacements = (
        ("lora_down", "lora_up"),
        ("lora_A", "lora_B"),
    )
    for source, target in replacements:
        if source in key:
            return key.replace(source, target)
    return key


def _orient_lora_matrices(
    down: np.ndarray,
    up: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return matrices in an unambiguous U @ D orientation."""
    if down.ndim != 2 or up.ndim != 2:
        return None

    # The lora_down/lora_A and lora_up/lora_B key names already tell us
    # which tensor is which; the standard PEFT-style convention stores
    # down as (rank, in_dim) and up as (out_dim, rank). Trust that layout
    # first rather than guessing from shape alone -- guessing across all
    # four transposed combinations is ambiguous whenever in_dim == out_dim
    # (e.g. square attention projections), which caused legitimate,
    # correctly-oriented pairs to be rejected as "ambiguous".
    if up.shape[1] == down.shape[0]:
        return down, up

    # Fall back for files that store the factors in the reverse layout
    # (down as (in_dim, rank), up as (rank, out_dim)).
    if up.shape[0] == down.shape[1]:
        return down.T, up.T

    return None


def _lora_singular_values(down: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Compute non-zero singular values from a small LoRA core.

    If U is (out x r) and D is (r x in), QR decompositions give
        U = Qu Ru
        D = Rd.T Qd.T
    and U@D has exactly the same non-zero singular values as
        Ru @ Rd.T, which is only (r x r).

    This avoids materializing a potentially enormous out x in matrix and avoids
    a full SVD of that matrix—the main source of the apparent 'hang'.
    """
    # The factor rank is the only dimension that should drive this computation.
    if up.shape[1] != down.shape[0]:
        raise ValueError(f"Incompatible LoRA shapes: up={up.shape}, down={down.shape}")

    # QR on the tall factors is cheap when r is small, which is the normal LoRA case.
    _, r_up = np.linalg.qr(up, mode="reduced")
    _, r_down_t = np.linalg.qr(down.T, mode="reduced")
    core = r_up @ r_down_t.T
    return svdvals(core)


def check_pair(
    file_path: str,
    down_key: str,
    up_key: str,
    index: dict[str, dict[str, Any]],
    max_rank: int = 1024,
    max_dense_elements: int = 4_000_000,
    spectral_threshold: float = DEFAULT_SPECTRAL_THRESHOLD,
) -> dict[str, str] | None:
    """Inspect one LoRA matrix pair without forming the full weight update."""
    try:
        down = _numpy_tensor(file_path, down_key, index)
    except Exception as exc:
        return result(
            "LoRA spectral",
            "error",
            f"{down_key}: {type(exc).__name__}: {exc}",
        )

    if down.ndim != 2:
        return None

    if up_key not in index:
        # For a lone LoRA factor, avoid expensive full SVDs of giant matrices.
        if down.size > max_dense_elements:
            return result(
                "LoRA spectral",
                "skipped",
                f"{down_key}: unpaired matrix too large for dense spectral check",
            )
        try:
            singular_values = svdvals(down)
        except Exception as exc:
            return result(
                "LoRA spectral",
                "error",
                f"{down_key}: SVD failed: {type(exc).__name__}: {exc}",
            )
    else:
        try:
            up = _numpy_tensor(file_path, up_key, index)
        except Exception as exc:
            return result(
                "LoRA spectral",
                "error",
                f"{up_key}: {type(exc).__name__}: {exc}",
            )

        if up.ndim != 2:
            return None

        oriented = _orient_lora_matrices(down, up)
        if oriented is None:
            return result(
                "LoRA spectral",
                "error",
                f"{down_key}: ambiguous/incompatible shapes down={down.shape}, up={up.shape}",
            )

        down, up = oriented
        rank = min(down.shape[0], up.shape[1])
        if rank <= 0:
            return result("LoRA spectral", "clean", f"{down_key}: empty factor")
        if rank > max_rank:
            return result(
                "LoRA spectral",
                "skipped",
                f"{down_key}: LoRA rank {rank} exceeds limit {max_rank}",
            )

        try:
            singular_values = _lora_singular_values(down, up)
        except Exception as exc:
            return result(
                "LoRA spectral",
                "error",
                f"{down_key}: spectral core failed: {type(exc).__name__}: {exc}",
            )

    if not np.isfinite(singular_values).all():
        return result(
            "LoRA spectral",
            "malicious",
            f"{down_key}: non-finite numeric values",
        )

    if singular_values.size == 0:
        return result("LoRA spectral", "clean", f"{down_key}: empty spectrum")

    total = float(np.sum(singular_values, dtype=np.float64))
    top = float(singular_values[0])
    ratio = top / total if total > 0 else 0.0

    # This remains a heuristic anomaly detector, not proof of malware.
    status = "malicious" if ratio > spectral_threshold else "clean"
    return result(
        "LoRA spectral",
        status,
        f"{down_key}: top singular-value share={ratio:.1%}",
    )


def run_lora(
    file_path: str,
    max_rank: int = 1024,
    spectral_threshold: float = DEFAULT_SPECTRAL_THRESHOLD,
) -> dict[str, str]:
    """Run the LoRA spectral heuristic on safetensors files."""
    if Path(file_path).suffix.lower() not in SAFETENSORS_EXTS:
        return result("LoRA spectral", "skipped", "Not a safetensors file")

    try:
        index = _read_safetensors_index(file_path)
    except Exception as exc:
        return result(
            "LoRA spectral",
            "error",
            f"Header read failed: {type(exc).__name__}: {exc}",
        )

    down_keys = sorted(
        key for key in index if "lora_down" in key or "lora_A" in key
    )
    if not down_keys:
        return result("LoRA spectral", "skipped", "No LoRA matrices found")

    checks: list[dict[str, str]] = []
    for down_key in down_keys:
        check = check_pair(
            file_path,
            down_key,
            pair_key(down_key),
            index,
            max_rank=max_rank,
            spectral_threshold=spectral_threshold,
        )
        if check is not None:
            checks.append(check)

    if not checks:
        return result("LoRA spectral", "skipped", "No 2-D LoRA matrices found")

    malicious = [c for c in checks if c["status"] == "malicious"]
    errors = [c for c in checks if c["status"] == "error"]
    clean = [c for c in checks if c["status"] == "clean"]
    skipped = [c for c in checks if c["status"] == "skipped"]

    if malicious:
        return result(
            "LoRA spectral",
            "malicious",
            "; ".join(c["detail"] for c in malicious[:3]),
        )
    if errors and not clean:
        return result(
            "LoRA spectral",
            "error",
            "; ".join(c["detail"] for c in errors[:3]),
        )
    if errors:
        return result(
            "LoRA spectral",
            "error",
            f"{len(clean)} layer(s) passed; {len(errors)} layer(s) could not be analyzed",
        )
    if skipped and not clean:
        return result(
            "LoRA spectral",
            "skipped",
            f"{len(skipped)} LoRA layer(s) exceeded the spectral limits",
        )

    return result(
        "LoRA spectral",
        "clean",
        f"{len(clean)} layer(s) analyzed",
    )


def scan(file_path: str) -> dict[str, Any]:
    """Run all applicable checks against a file."""
    extension = Path(file_path).suffix.lower()
    checks: list[dict[str, str]] = []

    # Extension is only used to decide which scanners are applicable.
    # The scanners themselves still validate/parse the actual file.
    if extension in PICKLE_EXTS:
        checks.extend(
            [
                run_picklescan(file_path),
                run_fickling(file_path),
                run_modelscan(file_path),
            ]
        )
    else:
        checks.append(
            result(
                "pickle-layer",
                "skipped",
                "Extension is not a recognized pickle/model format",
            )
        )

    checks.append(run_lora(file_path))

    statuses = {check["status"] for check in checks}

    if "malicious" in statuses:
        overall_status = "malicious"
    elif "error" in statuses:
        overall_status = "error"
    elif "clean" in statuses:
        overall_status = "clean"
    else:
        overall_status = "skipped"

    stat = os.stat(file_path)

    return {
        "engine": "kavacha",
        "version": 2,
        "file": os.path.basename(file_path),
        "file_size": stat.st_size,
        "file_hash": sha256(file_path),
        "checks": checks,
        "overall_status": overall_status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AI Kavacha sidecar scanner for ML model/adaptor files."
    )
    parser.add_argument("path", help="Path to the model/adaptor file")
    args = parser.parse_args()

    file_path = os.path.abspath(args.path)

    if not os.path.isfile(file_path):
        print(
            json.dumps(
                {
                    "engine": "error",
                    "error": f"File not found: {args.path}",
                    "checks": [],
                },
                separators=(",", ":"),
            )
        )
        return 1

    try:
        report = scan(file_path)
    except Exception as exc:
        report = {
            "engine": "kavacha",
            "file": os.path.basename(file_path),
            "checks": [],
            "overall_status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(report, separators=(",", ":")))
        return 2

    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    # Explicitly select "spawn" where available to avoid inheriting large
    # parent-process state in long-running hosts.
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    raise SystemExit(main())
