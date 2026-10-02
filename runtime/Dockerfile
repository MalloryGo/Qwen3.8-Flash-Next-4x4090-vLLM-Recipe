# Build from repository root:
# docker build --pull=false -t qwen38-merlin-vllm:0.30.0 runtime/
#
# Thin Python overlay only; no vLLM, PyTorch or CUDA reinstall/rebuild.
FROM vllm/vllm-openai:v0.30.0-ubuntu2404@sha256:ad78b2a8b0eaa8f9da1b9a81dcf38395213c754d2dc795456da3e09b8cb327d2

ARG SOURCE_COMMIT=b21ea4582c8bd947c6c469bc495b9a73ea449893
LABEL org.opencontainers.image.source="https://github.com/MalloryGo/Qwen3.8-Flash-Next-4x4090-vLLM-Recipe" \
      org.opencontainers.image.version="v1.0.0" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}"

COPY overlay/vllm /tmp/qwen38-overlay/vllm
RUN python3 -c "import importlib.util, pathlib, shutil; spec=importlib.util.find_spec('vllm'); assert spec and spec.origin; dst=pathlib.Path(spec.origin).parent; shutil.copytree('/tmp/qwen38-overlay/vllm', dst, dirs_exist_ok=True); shutil.rmtree('/tmp/qwen38-overlay')"
