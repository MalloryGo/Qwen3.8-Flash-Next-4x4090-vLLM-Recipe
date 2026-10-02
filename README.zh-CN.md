# Qwen3.8-Flash-Next Merlin：4× RTX 4090 + vLLM 部署方案

**[English](README.md) | 简体中文 | [日本語](README.ja.md) | [한국어](README.ko.md)**

这是一个可以直接照着跑的 **4× RTX 4090 24 GB（SM89）** 部署方案，用
**vLLM 0.30.0** 运行
[halt95/Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin)。

项目只做几件事：

- 从原模型仓库下载权重；
- 基于官方 vLLM 0.30.0 构建一个很薄的兼容层；
- 用现成脚本启动服务；
- 使用 262,144 token 上下文、MTP2、FP8 KV、TP2×PP2、EP；
- 提供 OpenAI 兼容的 Chat Completions 和 `/v1/responses`；
- 推荐同时 active 的推理请求数为 **4**，更多请求交给队列等待。

这不是模型发布，也不是 vLLM fork。仓库**不包含模型权重**。

## 已验证配置

| 项目 | 配置 |
|---|---|
| GPU | 4× NVIDIA RTX 4090 24 GB，SM89 |
| 系统内存 | 256 GB DDR4-3200 ECC RDIMM |
| 系统 | Ubuntu 24.04 |
| vLLM | 0.30.0，commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| 并行方式 | TP2 × PP2，PP 分层 `25,23`，EP 开启 |
| 最大上下文 | `262144` |
| MTP | 2 个 speculative tokens |
| KV Cache | FP8 E4M3 |
| PLE | CPU pinned memory / UVA |
| Prefix Cache | 默认开启；附带关闭版本 |
| Active 并发 | **4** |
| API | Chat Completions + `/v1/responses` |
| 多模态 | 图片、视频均已验证 |

不同机器的 GPU 编号和拓扑可能不一样。脚本默认使用 `0,1,2,3`，如果你的拓扑更适合其他顺序，可以通过 `GPU_ORDER` 覆盖。

## 1. 环境要求

本方案在 Ubuntu 24.04 + Docker + NVIDIA Container Toolkit 上验证。

建议准备：

- 4 张 RTX 4090 24 GB；
- 可正常使用 GPU 的 NVIDIA 驱动和 Docker；
- 系统内存：测试配置为 256 GB DDR4-3200 ECC RDIMM；这套测试 workload 的 host memory 峰值约 89.4 GiB；
- 模型权重大约需要 120 GiB 磁盘空间，另外还要预留 Docker 镜像空间；
- 首次下载模型和基础镜像时需要联网。

先确认 Docker 能看到 GPU：

~~~bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
~~~

## 2. 下载模型

模型原始地址：

**https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin**

本方案验证时使用的 checkpoint revision：

~~~text
483ee0015419568c7014b011db0c08b6ddcb2ceb
~~~

使用 Hugging Face 当前的 `hf` CLI（如果系统里没有 `hf` 命令，先安装或更新 `huggingface_hub`）：

~~~bash
mkdir -p /srv/qwen38/model

hf download halt95/Qwen3.8-Flash-Next-W4A16-Merlin \
  --revision 483ee0015419568c7014b011db0c08b6ddcb2ceb \
  --local-dir /srv/qwen38/model
~~~

下载完成后，模型目录至少应包含这类文件：

~~~text
config.json
model.safetensors.index.json
tokenizer*
processor / preprocessor 文件
qsa_kv_scales_262k.json
27 个 .safetensors 权重分片
~~~

模型权重、tokenizer 和 QSA sidecar 仍按原模型仓库的许可证使用，本仓库不重新分发这些文件。

## 3. 下载本项目

~~~bash
git clone https://github.com/MalloryGo/Qwen3.8-Flash-Next-4x4090-vLLM-Recipe.git
cd Qwen3.8-Flash-Next-4x4090-vLLM-Recipe
~~~

## 4. 构建运行镜像

正常使用时**不需要自己手工给 vLLM 打 patch**。

仓库里的 Dockerfile 直接以固定版本的官方 vLLM 0.30.0 镜像为基础，再覆盖 10 个已经整理好的 Python 文件：

~~~bash
docker build --pull=false \
  -t qwen38-merlin-vllm:0.30.0 \
  runtime/
~~~

对大多数人来说，这一步就是全部的 runtime 构建工作，不需要自己手工打 patch。构建过程不会重新安装或重新编译 CUDA、PyTorch、vLLM。

7 个源码改动同时保存在 `runtime/patches/`，主要用于审计、追溯和以后迁移到新版本。详细说明见 [docs/patches.md](docs/patches.md)。

## 5. 创建 API Key

~~~bash
sudo mkdir -p /srv/qwen38
openssl rand -hex 32 | sudo tee /srv/qwen38/vllm-api-key >/dev/null
sudo chmod 600 /srv/qwen38/vllm-api-key
~~~

## 6. 启动服务

在仓库根目录执行：

启动脚本本身就是一层很薄的 `docker run` 封装，完整容器参数都在脚本里。这样 README 不再重复一份参数，后面也不容易出现两边不一致。

~~~bash
export MODEL_DIR=/srv/qwen38/model
export CONFIG_DIR="$PWD/configs"
export API_KEY_FILE=/srv/qwen38/vllm-api-key
export RUNTIME_IMAGE=qwen38-merlin-vllm:0.30.0

bash configs/serve-262k-mtp2-prefix-on.sh
~~~

默认参数：

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

默认只监听 `127.0.0.1:8000`。

如果你的 GPU 拓扑不同，可以覆盖默认 GPU 顺序：

~~~bash
export GPU_ORDER=0,1,2,3
~~~

如果希望关闭 Prefix Cache：

~~~bash
bash configs/serve-262k-mtp2-prefix-off.sh
~~~

## 7. 检查 API

~~~bash
docker ps
docker logs --tail=100 qwen38-prefix-on

API_KEY="$(cat /srv/qwen38/vllm-api-key)"

curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $API_KEY"
~~~

Chat Completions 示例：

~~~bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-flash","messages":[{"role":"user","content":"只回复 OK"}],"max_tokens":32}'
~~~

Responses API 示例：

~~~bash
curl http://127.0.0.1:8000/v1/responses \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-flash","input":"只回复 OK"}'
~~~

## 8. 这 7 个 patch 做了什么

最终运行环境是官方 vLLM 0.30.0，加上 7 个范围很小、固定版本的兼容改动：

1. 回移 upstream 的 Qwen4Exp FP8 QSA/KV 原生路径；
2. 修复 Pipeline Parallel 下空 projected KV group 的问题；
3. 修复 hybrid cache 场景中 PP worker 的 KV layout 交集；
4. 当全部 PLE layer 都位于 PP0 时允许 PLE + PP；
5. 修复最终 PP stage 上 Qwen4Exp MTP drafter 的输入归属判断；
6. 加载并校验 Merlin 的 `qsa_kv_scales_262k.json`，并按本地 PP layer 应用；
7. 明确把 host KV scale 副本创建在 CPU 上。

普通使用者直接构建 `runtime/Dockerfile` 即可，不需要手工应用 patch。

如果希望自己从源码复现：

~~~bash
git clone https://github.com/vllm-project/vllm.git vllm-src
cd vllm-src
git checkout ced6857afa0ea7b2e3f0846a62e1394e90f15607

git am ../runtime/patches/*.patch
~~~

具体来源、commit 和 hash 见 [runtime/patch_manifest.json](runtime/patch_manifest.json)。

## 9. Benchmark

下面的数字只代表这台测试机器，作为参考即可，不应理解为其他机器一定能得到相同结果。单请求 decode rate 按 **1000 / median TPOT(ms)** 计算，表示生成阶段的 token 间隔速度，不等同于整条请求的端到端吞吐。

| Prompt | Median TTFT | Median TPOT | Decode |
|---:|---:|---:|---:|
| 128K | 10.88 s | 4.860 ms | **205.8 tok/s** |
| 220K | 19.12 s | 4.837 ms | **206.7 tok/s** |
| 255K | 22.31 s | 5.191 ms | **192.6 tok/s** |
| 260K | 22.74 s | 4.837 ms | **206.8 tok/s** |

同时验证过总长度正好 **262,144 token** 的请求：261,888 prompt + 256 output。

并发数据使用整组请求的 aggregate output throughput：

| 场景 | 完成 | Aggregate output throughput |
|---|---:|---:|
| 2 × 128K | 2/2 | 38.4 tok/s |
| 3 × 80K | 3/3 | 53.8 tok/s |
| 4 × 60K | 4/4 | **75.8 tok/s** |
| 混合 100K/60K/40K/20K | 4/4 | **85.8 tok/s** |
| 12 请求突发，最多 active 4 | 12/12 | **95.3 tok/s** |

MTP 对比：

| 模式 | Median decode | KV pool |
|---|---:|---:|
| MTP off | 100.18 tok/s | — |
| MTP2 | **167.41 tok/s** | 264,527 tokens |
| MTP3 | 169.79 tok/s | 248,584 tokens |

该组测试中 MTP3 只比 MTP2 快约 1.42%，但 MTP2 多保留 15,943 个 KV token，因此最终推荐 MTP2。

重复约 128K 前缀时，Prefix Cache 将 TTFT 从约 **13.93 s** 降到 **0.85 s**，命中 124,176 / 128,000 prompt tokens。

另外已经验证：

- `/v1/responses`
- streaming
- tool calling
- tool result 后继续生成
- JSON object / strict JSON Schema
- 图片输入
- 视频输入
- 12 路 Agent 同时到达、最多 active 4，`tool_choice=auto` 时 12/12 正常完成

机器可读结果见 [benchmarks/results.json](benchmarks/results.json)。

## 10. 已知限制

并发使用 **forced/required tool choice** 时，XGrammar 偶尔会返回 HTTP 500。

建议：

- 普通并发 Agent：使用 `tool_choice=auto`；
- forced/required：暂时限制为一次只运行一个请求。

其他注意事项见 [docs/known-issues.md](docs/known-issues.md)。

## 目录结构

~~~text
runtime/      Dockerfile、thin overlay、7 个 patch 和 manifest
configs/      262K / MTP2 / active-4 启动配置