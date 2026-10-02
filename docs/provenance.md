# Sources and attribution

This repository is an independent deployment recipe. It does not redistribute
model weights, Docker image archives or third-party media.

## Model

- halt95/Qwen3.8-Flash-Next-W4A16-Merlin
- https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin
- validated checkpoint revision:
  `483ee0015419568c7014b011db0c08b6ddcb2ceb`
- license: use the license stated by the original model repository

## vLLM

- https://github.com/vllm-project/vllm
- v0.30.0
- base commit: `ced6857afa0ea7b2e3f0846a62e1394e90f15607`
- license: Apache-2.0

Patches 1–3 are upstream/backported vLLM changes and link to the corresponding
upstream PR/commit in `runtime/patch_manifest.json`.

Patches 4–7 are small integration changes needed for this serving layout.

## Public references consulted during development

- https://github.com/halt95/qwen38-flash-next-3090s
- https://github.com/jon-nielsen/qwen38-flash-next-3090-kit

These projects were useful references for model/runtime behavior and deployment
comparison. Their complete runtime trees, SM86-specific binaries/workarounds and
model files are not copied into this repository.

See [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for the consolidated
third-party notice.
