"""
Generates test-lora-spike.safetensors — a LoRA adapter whose down/up
matrices are engineered to be numerically concentrated (rank-1-like),
which is exactly the pattern the LoRA spectral heuristic in
scan_engine.py flags as "malicious" (top singular-value share > 0.85).

There is no code-execution payload here at all — safetensors is a pure
data format. This only demonstrates the NUMERIC pattern the heuristic
is designed to catch, so you can confirm the "Weight-space layer"
correctly fires.

For comparison it also writes test-lora-normal.safetensors, a LoRA
with singular values spread across several directions (typical of a
normal, broadly-trained adapter) which should come back "clean" —
useful for spotting false positives.
"""
import numpy as np
from safetensors.numpy import save_file

RANK = 16
OUT_DIM = 64
IN_DIM = 64


def make_spiked_lora(rank=RANK, out_dim=OUT_DIM, in_dim=IN_DIM, seed=0):
    """Construct down/up matrices whose product is (numerically) rank-1
    dominant: one direction carries almost all the singular-value mass."""
    rng = np.random.default_rng(seed)

    # Build an explicitly rank-1-dominant core, then embed it via random
    # orthonormal bases so it isn't trivially "just a rank-1 matrix" —
    # it still looks like an ordinary-shaped LoRA pair.
    u_dom = rng.normal(size=(out_dim, 1)).astype(np.float32)
    v_dom = rng.normal(size=(1, rank)).astype(np.float32)
    dominant = u_dom @ v_dom  # out_dim x rank, effectively rank 1

    noise = rng.normal(size=(out_dim, rank)).astype(np.float32) * 0.01
    up = dominant + noise  # out_dim x rank

    down = rng.normal(size=(rank, in_dim)).astype(np.float32)
    down /= np.linalg.norm(down, axis=1, keepdims=True)

    return down.astype(np.float32), up.astype(np.float32)


def make_normal_lora(rank=RANK, out_dim=OUT_DIM, in_dim=IN_DIM, seed=1):
    """A LoRA pair with singular value mass spread across several
    directions — representative of a typical, broadly trained adapter."""
    rng = np.random.default_rng(seed)
    down = rng.normal(size=(rank, in_dim)).astype(np.float32) * 0.05
    up = rng.normal(size=(out_dim, rank)).astype(np.float32) * 0.05
    return down, up


def write_lora_file(path, down, up, layer="base_model.model.model.layers.0.self_attn.q_proj"):
    tensors = {
        f"{layer}.lora_down.weight": down,
        f"{layer}.lora_up.weight": up,
    }
    save_file(tensors, path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    down, up = make_spiked_lora()
    write_lora_file("test-lora-spike.safetensors", down, up)

    down_n, up_n = make_normal_lora()
    write_lora_file("test-lora-normal.safetensors", down_n, up_n)
