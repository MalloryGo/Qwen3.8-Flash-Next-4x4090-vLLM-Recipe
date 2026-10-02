# Third-party notices and credits

This repository is a serving/deployment recipe. It does **not** redistribute model
weights, Docker image archives, or third-party media.

## vLLM

- Project: https://github.com/vllm-project/vllm
- Pinned base: v0.30.0, commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`
- License: Apache License 2.0

The `runtime/overlay/vllm/` files are modified vLLM source files and retain vLLM's
SPDX/copyright headers. Patches 1-3 are traceable to upstream vLLM PRs/commits in
`runtime/patch_manifest.json`.

## Merlin checkpoint

- Checkpoint: https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin
- Base family: Qwen3.8-Flash-Next
- License reported by the checkpoint/model card: Qwen Community License 1.0

No model weights, tokenizer files, QSA sidecar, or model media are hosted here.
Users must obtain the checkpoint separately and comply with its license.

The Merlin model card credits Qwen, Intel/AutoRound, RadixArk, DominikBucko and
community vLLM work for components of the checkpoint/serving lineage. Refer to
the model card for the authoritative lineage and license notices.

## Reference serving repositories

The implementation and deployment investigation also compared against these
public serving recipes:

- https://github.com/halt95/qwen38-flash-next-3090s — Apache-2.0
- https://github.com/jon-nielsen/qwen38-flash-next-3090-kit — Apache-2.0

Their complete runtime trees, SM86-specific binaries/workarounds, and model
artifacts are not included in this repository. Where this recipe uses upstream
vLLM commits, the original vLLM PR/commit is linked directly in the patch manifest.

## No endorsement

This repository is an independent community recipe. It is not produced, reviewed,
or endorsed by Qwen, halt95, vLLM, Intel, RadixArk, or the maintainers of the
reference repositories above.
