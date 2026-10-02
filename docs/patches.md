# Runtime patches

Base runtime:

- repository: https://github.com/vllm-project/vllm
- version: v0.30.0
- commit: `ced6857afa0ea7b2e3f0846a62e1394e90f15607`

Normal users do not need to apply these patches manually. The supplied
`runtime/Dockerfile` copies the already prepared thin Python overlay onto the
pinned official vLLM image.

The patch files are kept so the changes remain easy to audit and port.

## 1. Native Qwen4Exp FP8 QSA/KV path

Backport of upstream vLLM PR #55557 / commit
`dff1bde84dd6e34a49c150116d0f212507280910`.

This supplies the native FP8 E4M3 QSA path used on SM89.

## 2. Empty projected KV groups under PP

Backport of upstream PR #54793.

Prevents empty projected uniform KV groups from allocating tensors for layers
owned by another pipeline stage.

## 3. PP KV-layout intersection

Backport of upstream PR #54795.

Uses the ordered intersection of KV-cache layouts when pipeline stages own
different hybrid-cache layers.

## 4. PLE ownership with pipeline parallelism

The upstream Qwen4Exp gate rejected PP whenever PLE was enabled. This recipe
allows PP only when every configured PLE layer belongs to PP0. With the tested
25/23 partition, the PLE layer is on PP0.

## 5. MTP drafter input ownership

On the final pipeline stage, the Qwen4Exp MTP drafter must select its input based
on whether `intermediate_tensors` are present rather than `is_first_rank`.

## 6. Merlin QSA KV-scale sidecar

Adds validation and loading for `qsa_kv_scales_262k.json`:

- exact global QSA-layer coverage;
- finite positive K/V scale values;
- application only to the QSA layers local to each PP stage.

## 7. Keep host KV-scale copies on CPU

Model construction can occur under a CUDA default-device context. The documented
host copies `_k_scale_cpu` and `_v_scale_cpu` therefore use explicit
`device="cpu"`.

## Manual reproduction

~~~bash
git clone https://github.com/vllm-project/vllm.git vllm-src
cd vllm-src
git checkout ced6857afa0ea7b2e3f0846a62e1394e90f15607

git am ../runtime/patches/*.patch
~~~

Exact patch hashes and provenance are recorded in
[runtime/patch_manifest.json](../runtime/patch_manifest.json).

When moving to a newer vLLM release, first check which changes have already landed
upstream. Do not blindly reapply all seven patches.