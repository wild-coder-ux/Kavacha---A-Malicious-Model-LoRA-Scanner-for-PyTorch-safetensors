
# Kavacha — Malicious Model & LoRA Scanner for PyTorch/safetensors

A local, defensive scanner for ML model and adaptor files (`.bin`, `.pt`, `.pth`, `.ckpt`, `.pkl`, `.pickle`, `.safetensors`). It never deserializes the target model in-process, runs pickle-focused scanners in isolated subprocesses, and inspects LoRA `safetensors` adaptors numerically to catch spiked/rank-1-dominant weight patterns without loading the full model into memory.

## Design goals

- **Never deserialize the target model** inside the main process.
- **Isolate pickle-focused scanners** (picklescan) in their own subprocess/process with a hard timeout.
- **Hash every file** before returning a result.
- **Inspect LoRA safetensors numerically** — via lightweight QR + SVD on the small `rank × rank` core, not a full `out_dim × in_dim` reconstruction — without loading the full model.
- **Fail closed** — scanner errors are reported as explicit `"error"` checks, never silently treated as `"clean"`.

## What it checks

| Layer | Applies to | What it does |
|---|---|---|
| `picklescan` | pickle-family extensions | Runs [picklescan](https://github.com/mmaitre314/picklescan) in an isolated subprocess and reports flagged globals. |
| `fickling` | pickle-family extensions | Runs [Fickling](https://github.com/trailofbits/fickling) and distinguishes a tool error from a genuine malicious finding. |
| `modelscan` | pickle-family extensions | Runs [ModelScan](https://github.com/protectai/modelscan) and parses its output for severity markers. |
| `LoRA spectral` | `.safetensors` | Reads `lora_down`/`lora_up` (or `lora_A`/`lora_B`) tensor pairs, computes singular values of the low-rank core, and flags adaptors where one direction carries an outsized share of the spectral mass (default threshold: 85%). |

Each check returns one of `clean`, `malicious`, `suspicious`, `error`, or `skipped`. The overall file status is `malicious` if any check is malicious, else `error` if any check errored, else `clean` if any check is clean, else `skipped`.

## Files

- **`scan_engine.py`** — the scanner itself. Run it directly against a model/adaptor file.
- **`gensafetenor.py`** — generates two test `safetensors` LoRA adaptors:
  - `test-lora-spike.safetensors` — engineered to be rank-1-dominant (top singular-value share > 85%), the numeric pattern the spectral heuristic is designed to catch. **This is a pure data file with no code-execution payload** — `safetensors` is a data-only format.
  - `test-lora-normal.safetensors` — a typical adaptor with singular-value mass spread across several directions, used to check for false positives.

## Requirements

```bash
pip install numpy scipy safetensors
```

Optional, for full pickle-layer coverage:

```bash
pip install picklescan fickling modelscan
```

Any of these that aren't installed are reported as `"error"` (`... is not installed or not on PATH`) rather than silently skipped, in keeping with the fail-closed design.

## Usage

Generate the test fixtures:

```bash
python3 gensafetenor.py
```

Scan a file:

```bash
python3 scan_engine.py test-lora-spike.safetensors
python3 scan_engine.py test-lora-normal.safetensors
```

Each run prints a single JSON report to stdout, e.g.:

```json
{
  "engine": "kavacha",
  "version": 2,
  "file": "test-lora-spike.safetensors",
  "file_size": 8456,
  "file_hash": "ad0de6216de429233e6f64e96f6f741bbd8b503e983589d56df9db25fb05b686",
  "checks": [
    {"label": "pickle-layer", "status": "skipped", "detail": "Extension is not a recognized pickle/model format"},
    {"label": "LoRA spectral", "status": "malicious", "detail": "...lora_down.weight: top singular-value share=97.1%"}
  ],
  "overall_status": "malicious"
}
```

Exit codes: `0` on a completed scan (check `overall_status` for the verdict), `1` if the file doesn't exist, `2` on an unhandled scanner error.

## Running the app (Tauri dev mode)

The scanner is bundled as a sidecar for a Tauri desktop app. To run the app in development mode:

```bash
npm install
npm run tauri dev
```

`npm install` only needs to be run once (or whenever `package.json` changes) — it populates `node_modules/`, which is gitignored and not part of the repo. `npm run tauri dev` starts the frontend dev server and launches the Tauri window, which talks to the `scan_engine.py` sidecar under `src-tauri/sidecar/`.

## Notes on the LoRA orientation check

`lora_down`/`A` tensors follow the standard PEFT convention of shape `(rank, in_dim)`, and `lora_up`/`B` tensors are `(out_dim, rank)` — the key names already make the orientation unambiguous. The engine trusts that convention directly, with a fallback for adaptors stored in the reverse layout. (An earlier version tried to re-derive orientation purely from shape by testing all four transposed combinations, which produced false "ambiguous shape" errors whenever `in_dim == out_dim`, e.g. on same-dimension attention projections like `q_proj`.)

## Caveats

- The spectral heuristic is an **anomaly detector, not proof of malicious intent**. A `malicious` verdict on the LoRA layer means the weights are numerically unusual (spiked/rank-1-dominant), which is worth a human look — not a guaranteed exploit.
- Large or high-rank LoRA factors are skipped from the dense spectral check (`max_rank`, `max_dense_elements`) to avoid expensive computation; a `skipped` result there is not the same as `clean`.
