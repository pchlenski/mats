import torch

MODEL = "gelu-1l"
RUN = "run1"
DATASET = "NeelNanda/c4-code-20k"
DD = f"/root/mats-sprint/data/processed/{MODEL}_{RUN}_audit"

BATCH_SIZE = 128

# Neel vars
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

SPACE = "·"
NEWLINE = "↩"
TAB = "→"


# Defining the autoencoder
SAE_CFG = {
    "seed": 49,
    "batch_size": 4096,
    "buffer_mult": 384,
    "lr": 1e-4,
    "num_tokens": int(2e9),
    "l1_coeff": 3e-4,
    "beta1": 0.9,
    "beta2": 0.99,
    "dict_mult": 8,
    "seq_len": 128,
    "d_mlp": 2048,
    "enc_dtype": "fp32",
    "remove_rare_dir": False,
}
SAE_CFG["model_batch_size"] = 64
SAE_CFG["buffer_size"] = SAE_CFG["batch_size"] * SAE_CFG["buffer_mult"]
SAE_CFG["buffer_batches"] = SAE_CFG["buffer_size"] // SAE_CFG["seq_len"]
