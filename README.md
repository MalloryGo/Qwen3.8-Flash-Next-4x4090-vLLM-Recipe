# Qwen3.8-Flash-Next Merlin on 4× RTX 4090 with vLLM

**English | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md)**

A working deployment recipe for
[halt95/Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin)
on **4× RTX 4090 24 GB (SM89)** with **vLLM 0.30.0**.

If your setup is close to this one, the shortest path is:

- download the model from the original model repository;
- build the included thin vLLM overlay;
- start the server with the supplied launch script;
- use a 262,144-token context window, MTP2, FP8 KV cache, TP2×PP2 and EP;
- expose OpenAI-compatible Chat Completions and `/v1/responses`;
- keep active inference concurrency at **4**, and queue larger bursts.

This is a deployment recipe, **not** a model release and **not** a vLLM fork. Model weights are not redistributed here.

## Tested configuration

| Item | Tested value |
|---|---|
| GPU | 4× NVIDIA RTX 4090 24 GB, SM89 |
| OS | Ubuntu 24.04 |
| vLLM base | 0.30.0, commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| Parallelism | TP2 × PP2, PP partition `25,23`, EP enabled |
| Context | `262144` |
| MTP | 2 speculative tokens |
| KV cache | FP8 E4M3 |
| PLE | pinned CPU memory / UVA |
| Prefix cache | ON by default; OFF profile included |
| Active concurrency | **4** |
| API | Chat Completions + `/v1/responses` |
| Multimodal | image + video validated |

GPU numbering and topology vary by host. The launch script defaults to `0,1,2,3`; override `GPU_ORDER` if your topology benefits from a different order.

## 1. Prerequisites

The recipe was validated on Ubuntu 24.04 with Docker and NVIDIA Container Toolkit. You need:

- four 24 GB RTX 4090 GPUs;
- a working NVIDIA driver and Docker GPU runtime;
- ample system RAM; PLE uses CPU-pinned memory, but this recipe does not claim a minimum RAM requirement;
- roughly 120 GiB of storage for the model checkpoint, plus Docker image space;
- Internet access for the initial model/image download.

Verify Docker can see the GPUs:

~~~bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
~~~

## 2. Download the model

Original model repository:

**https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin**

Validated checkpoint revision:

~~~text
483ee0015419568c7014b011db0c08b6ddcb2ceb
~~~

Using the current Hugging Face CLI (install/update `huggingface_hub` first if the `hf` command is not available):

~~~bash
mkdir -p /srv/qwen38/model

hf download halt95/Qwen3.8-Flash-Next-W4A16-Merlin \
  --revision 483ee0015419568c7014b011db0c08b6ddcb2ceb \
  --local-dir /srv/qwen38/model
~~~

The model directory should include the checkpoint shards plus files such as:

~~~text
config.json
model.safetensors.index.json
tokenizer*
processor / preprocessor files
qsa_kv_scales_262k.json
27 × .safetensors shards
~~~

The weights, tokenizer and QSA sidecar remain under the terms stated by the original model repository.

## 3. Clone this recipe

~~~bash
git clone https://github.com/MalloryGo/Qwen3.8-Flash-Next-4x4090-vLLM-Recipe.git
cd Qwen3.8-Flash-Next-4x4090-vLLM-Recipe
~~~

## 4. Build the runtime image

The recommended path does **not** require manually patching vLLM. The Dockerfile starts from the pinned official vLLM 0.30.0 image and copies the included 10-file Python overlay.

~~~bash
docker build --pull=false \
  -t qwen38-merlin-vllm:0.30.0 \
  runtime/
~~~

For most users, this is the only runtime build step. The build does not reinstall or rebuild CUDA, PyTorch or vLLM. The seven source changes behind the overlay are kept in `runtime/patches/` for auditability. See [docs/patches.md](docs/patches.md).

## 5. Create an API key

~~~bash
sudo mkdir -p /srv/qwen38
openssl rand -hex 32 | sudo tee /srv/qwen38/vllm-api-key >/dev/null
sudo chmod 600 /srv/qwen38/vllm-api-key
~~~

## 6. Start the server

From the repository root:

The launch script is a thin `docker run` wrapper. The complete container parameters live in the script itself, so there is only one copy of the serving configuration to keep in sync.

~~~bash
export MODEL_DIR=/srv/qwen38/model
export CONFIG_DIR="$PWD/configs"
export API_KEY_FILE=/srv/qwen38/vllm-api-key
export RUNTIME_IMAGE=qwen38-merlin-vllm:0.30.0

bash configs/serve-262k-mtp2-prefix-on.sh
~~~

Default serving profile:

~~~text
TP = 2
PP = 2
PP layers = 25,23
EP = ON
MTP = 2
max_model_len = 262144
max_num_seqs = 4
max_num_batched_tokens = 1024
gpu_memory_utilization = 0.94
KV cache = fp8_e4m3
PLE = CPU pinned/UVA
prefix cache = ON
NCCL_P2P_DISABLE = 1
custom all-reduce = OFF
~~~

The API binds to `127.0.0.1:8000` by default.

If your GPU topology differs, override the tested order:

~~~bash
export GPU_ORDER=0,1,2,3
~~~

To disable prefix caching while keeping the rest of the profile unchanged:

~~~bash
bash configs/serve-262k-mtp2-prefix-off.sh
~~~

## 7. Check the API

~~~bash
docker ps
docker logs --tail=100 qwen38-prefix-on

API_KEY="$(cat /srv/qwen38/vllm-api-key)"

curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $API_KEY"
~~~

Chat Completions:

~~~bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-flash","messages":[{"role":"user","content":"Reply with: OK"}],"max_tokens":32}'
~~~

Responses API:

~~~bash
curl http://127.0.0.1:8000/v1/responses \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-flash","input":"Reply with: OK"}'
~~~

## 8. What the seven patches do

The runtime is official vLLM 0.30.0 plus seven small, version-pinned changes:

1. backport the native Qwen4Exp FP8 QSA/KV path from upstream vLLM;
2. fix empty projected KV groups under pipeline parallelism;
3. fix PP worker KV-layout intersection for hybrid cache layouts;
4. allow PLE with PP when all PLE layers are owned by PP0;
5. fix Qwen4Exp MTP drafter input ownership on the final PP stage;
6. load and validate Merlin's `qsa_kv_scales_262k.json` per local PP stage;
7. explicitly keep the documented host KV-scale copies on CPU.

Normal users should simply build `runtime/Dockerfile`. The patch files are included for review, provenance and future porting.

Manual source reproduction:

~~~bash
git clone https://github.com/vllm-project/vllm.git vllm-src
cd vllm-src
git checkout ced6857afa0ea7b2e3f0846a62e1394e90f15607

git am ../runtime/patches/*.patch
~~~

See [runtime/patch_manifest.json](runtime/patch_manifest.json) and [docs/patches.md](docs/patches.md).

## 9. Benchmark results

These numbers are from one test host, so treat them as reference values rather than guarantees. Single-request figures use **derived decode rate = 1000 / median TPOT(ms)**; this measures decode cadence, not end-to-end throughput.

| Prompt | Median TTFT | Median TPOT | Derived decode |
|---:|---:|---:|---:|
| 128K | 10.88 s | 4.860 ms | **205.8 tok/s** |
| 220K | 19.12 s | 4.837 ms | **206.7 tok/s** |
| 255K | 22.31 s | 5.191 ms | **192.6 tok/s** |
| 260K | 22.74 s | 4.837 ms | **206.8 tok/s** |

A combined **262,144-token** request boundary also completed successfully (261,888 prompt + 256 output).

Concurrency results use aggregate output throughput for the complete request wave:

| Workload | Completed | Aggregate output throughput |
|---|---:|---:|
| 2 × 128K | 2/2 | 38.4 tok/s |
| 3 × 80K | 3/3 | 53.8 tok/s |
| 4 × 60K | 4/4 | **75.8 tok/s** |
| Mixed 100K/60K/40K/20K | 4/4 | **85.8 tok/s** |
| 12-request burst, max active 4 | 12/12 | **95.3 tok/s** |

MTP selection at ~128K:

| Mode | Median decode | KV pool |
|---|---:|---:|
| MTP off | 100.18 tok/s | — |
| MTP2 | **167.41 tok/s** | 264,527 tokens |
| MTP3 | 169.79 tok/s | 248,584 tokens |

MTP3 was only 1.42% faster in that comparison, while MTP2 retained 15,943 more KV tokens, so MTP2 is the recommended capacity-first profile.

Prefix-cache reuse on a repeated ~128K prompt reduced TTFT from about **13.93 s** to **0.85 s**, with 124,176/128,000 prompt tokens reported as cache hits.

Agent/API checks included `/v1/responses`, streaming, tool calls, tool-result round trips, JSON object/strict JSON Schema, images and video. A 12-request Agent burst with `tool_choice=auto` and max active concurrency 4 completed 12/12.

Full machine-readable results: [benchmarks/results.json](benchmarks/results.json).

## 10. Known limitation

Concurrent **forced/required** tool choice can intermittently hit an XGrammar HTTP 500 error. Concurrent `tool_choice=auto` passed the tested workloads.

Recommended policy:

- normal concurrent Agent traffic: `tool_choice=auto`;
- forced/required tool choice: serialize to one in-flight request.

See [docs/known-issues.md](docs/known-issues.md).

## Repository layout

~~~text
runtime/      Dockerfile, thin overlay, seven patches and patch manifest
configs/      262K / MTP2 / active-4 launch profiles
benchmarks/   benchmark notes and machine-readable results
docs/         architecture, patches, known issues and provenance
scripts/      benchmark and integrity helpers
~~~

## License and attribution

Repository-authored documentation, configuration and helper scripts are released under Apache-2.0. Modified vLLM source files retain their upstream notices and carry a modification notice.

Model weights are **not** redistributed. Obtain the checkpoint directly from halt95's model repository and follow the license stated there.

See [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [docs/provenance.md](docs/provenance.md).