import numpy as np
from safetensors.numpy import save_file

# Create a minimal LoRA structure
# Real LoRAs have lora_down and lora_up matrices
lora_down = np.random.randn(16, 64).astype(np.float32) * 0.1  # rank 16
lora_up = np.random.randn(64, 16).astype(np.float32) * 0.1

tensors = {
    "base_model.model.model.layers.0.self_attn.q_proj.lora_down.weight": lora_down,
    "base_model.model.model.layers.0.self_attn.q_proj.lora_up.weight": lora_up,
}

metadata = {
    "lora_alpha": "32",
    "lora_r": "16",
    "base_model_name_or_path": "test-model"
}

save_file(tensors, "/home/wildtrain/Downloads/test-lora.safetensors", metadata=metadata)
print("✅ Created test-lora.safetensors in Downloads folder")
