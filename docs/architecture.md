# Architecture

## Hardware assumptions

- 4× NVIDIA RTX 4090 24 GB (Ada / SM89)
- PCIe-only multi-GPU setup
- Ubuntu 24.04
- ample host RAM for the CPU-pinned PLE allocation; no minimum RAM requirement is claimed

## Serving layout

~~~text
4 GPUs
  ├─ pipeline stage 0: 25 transformer layers
  │    └─ contains the PLE layer
  └─ pipeline stage 1: 23 transformer layers

Within each pipeline stage:
  TP = 2
  EP = enabled

MTP = 2
KV cache = FP8 E4M3
PLE = pinned CPU memory / UVA
max_model_len = 262144
max_num_seqs = 4
max_num_batched_tokens = 1024
~~~

TP2×PP2 keeps the model within 24 GB per GPU while preserving enough KV capacity
for one full 262K request and practical 1–4 request concurrency.

The launch script defaults to `GPU_ORDER=0,1,2,3`. GPU numbering and topology are host-specific, so override it when needed.

## API

The service exposes OpenAI-compatible Chat Completions and `/v1/responses`.
Image and video inputs were also validated.

The launcher binds to loopback by default.