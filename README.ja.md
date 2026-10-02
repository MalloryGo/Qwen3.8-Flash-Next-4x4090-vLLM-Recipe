# Qwen3.8-Flash-Next Merlin：4× RTX 4090 + vLLM レシピ

**[English](README.md) | [简体中文](README.zh-CN.md) | 日本語 | [한국어](README.ko.md)**

4× RTX 4090 24 GB (SM89) で
[halt95/Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin)
を **vLLM 0.30.0** で動かすための再現可能なデプロイ手順です。

このリポジトリはモデル配布や vLLM fork ではありません。モデル重みは含みません。
基本構成は 262,144 context、TP2×PP2、EP、MTP2、FP8 E4M3 KV、CPU pinned/UVA PLE、active concurrency 4 です。

## 検証構成

| 項目 | 構成 |
|---|---|
| GPU | 4× RTX 4090 24 GB, SM89, no NVLink |
| OS | Ubuntu 24.04 |
| vLLM | 0.30.0 @ `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| Parallelism | TP2 × PP2, layers `25,23`, EP ON |
| Context | `262144` |
| MTP | 2 |
| KV | FP8 E4M3 |
| Prefix cache | ON (OFF profile あり) |
| Active concurrency | **4** |
| API | Chat Completions + `/v1/responses` |
| Multimodal | image + video |

GPU 番号と topology はホストごとに異なります。スクリプトの既定値は `0,1,2,3` で、必要なら `GPU_ORDER` で変更できます。

## セットアップ

### 1. モデルを取得

モデル:

**https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin**

検証した revision:

~~~text
483ee0015419568c7014b011db0c08b6ddcb2ceb
~~~

~~~bash
mkdir -p /srv/qwen38/model

hf download halt95/Qwen3.8-Flash-Next-W4A16-Merlin \
  --revision 483ee0015419568c7014b011db0c08b6ddcb2ceb \
  --local-dir /srv/qwen38/model
~~~

`model.safetensors.index.json`、27 個の safetensors shard、`qsa_kv_scales_262k.json` などが必要です。

### 2. このリポジトリを取得

~~~bash
git clone https://github.com/MalloryGo/Qwen3.8-Flash-Next-4x4090-vLLM-Recipe.git
cd Qwen3.8-Flash-Next-4x4090-vLLM-Recipe
~~~

### 3. Runtime image を build

通常は手動で patch を当てる必要はありません。Dockerfile が公式 vLLM 0.30.0 image に必要な thin overlay を適用します。

~~~bash
docker build --pull=false -t qwen38-merlin-vllm:0.30.0 runtime/
~~~

### 4. API key を作成

~~~bash
sudo mkdir -p /srv/qwen38
openssl rand -hex 32 | sudo tee /srv/qwen38/vllm-api-key >/dev/null
sudo chmod 600 /srv/qwen38/vllm-api-key
~~~

### 5. 起動

~~~bash
export MODEL_DIR=/srv/qwen38/model
export CONFIG_DIR="$PWD/configs"
export API_KEY_FILE=/srv/qwen38/vllm-api-key
export RUNTIME_IMAGE=qwen38-merlin-vllm:0.30.0

bash configs/serve-262k-mtp2-prefix-on.sh
~~~

デフォルトでは `127.0.0.1:8000` に bind します。

GPU topology が異なる場合:

~~~bash
export GPU_ORDER=0,1,2,3
~~~

Prefix cache を無効化する場合:

~~~bash
bash configs/serve-262k-mtp2-prefix-off.sh
~~~

### 6. API 確認

~~~bash
API_KEY="$(cat /srv/qwen38/vllm-api-key)"

curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $API_KEY"
~~~

Responses API:

~~~bash
curl http://127.0.0.1:8000/v1/responses \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-flash","input":"Reply with: OK"}'
~~~

## 7 patches

公式 vLLM 0.30.0 に対して、次の小さな互換修正を使います。

1. Qwen4Exp FP8 QSA/KV native path の upstream backport
2. PP の empty projected KV group 修正
3. hybrid cache の PP KV-layout intersection 修正
4. PLE layer が PP0 にある場合の PLE + PP 対応
5. final PP stage の MTP drafter input ownership 修正
6. Merlin `qsa_kv_scales_262k.json` の検証付き loader
7. host KV-scale copy を明示的に CPU に配置

通常利用では `runtime/Dockerfile` を build するだけです。詳細は [docs/patches.md](docs/patches.md)。

## Benchmark

| Prompt | Median TTFT | Median TPOT | Decode |
|---:|---:|---:|---:|
| 128K | 10.88 s | 4.860 ms | **205.8 tok/s** |
| 220K | 19.12 s | 4.837 ms | **206.7 tok/s** |
| 255K | 22.31 s | 5.191 ms | **192.6 tok/s** |
| 260K | 22.74 s | 4.837 ms | **206.8 tok/s** |

262,144 token の combined boundary (261,888 prompt + 256 output) も完走しました。

| Workload | Success | Aggregate output |
|---|---:|---:|
| 2 × 128K | 2/2 | 38.4 tok/s |
| 3 × 80K | 3/3 | 53.8 tok/s |
| 4 × 60K | 4/4 | **75.8 tok/s** |
| 100K/60K/40K/20K | 4/4 | **85.8 tok/s** |
| 12 arrivals, max active 4 | 12/12 | **95.3 tok/s** |

MTP2 は MTP3 よりわずかに遅い一方、KV capacity を 15,943 token 多く確保できるため推奨設定です。

`/v1/responses`、streaming、tool calling、strict JSON Schema、image、video を確認済みです。詳細値は [benchmarks/results.json](benchmarks/results.json)。

## Known limitation

並列の **forced/required tool choice** は XGrammar HTTP 500 を断続的に返すことがあります。

- 並列 Agent traffic: `tool_choice=auto`
- forced/required: 1 request ずつ serialize

詳細: [docs/known-issues.md](docs/known-issues.md)

## License

リポジトリ独自の文書・設定・helper script は Apache-2.0。vLLM 由来の変更ファイルは upstream notice を保持します。

モデル重みは配布しません。元モデルリポジトリから取得し、そのライセンスに従ってください。

[NOTICE](NOTICE) / [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) / [docs/provenance.md](docs/provenance.md)