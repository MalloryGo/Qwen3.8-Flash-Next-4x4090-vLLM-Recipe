# Qwen3.8-Flash-Next Merlin: 4× RTX 4090 + vLLM 레시피

**[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | 한국어**

**4× RTX 4090 24 GB (SM89)** 환경에서
[halt95/Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin)
을 **vLLM 0.30.0**으로 실행하기 위한 재현 가능한 배포 레시피입니다.

이 저장소는 모델 배포본이나 vLLM fork가 아닙니다. 모델 가중치는 포함하지 않습니다.
권장 구성은 262,144 context, TP2×PP2, EP, MTP2, FP8 E4M3 KV, CPU pinned/UVA PLE, active concurrency 4입니다.

## 검증 환경

| 항목 | 구성 |
|---|---|
| GPU | 4× RTX 4090 24 GB, SM89, no NVLink |
| OS | Ubuntu 24.04 |
| vLLM | 0.30.0 @ `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| Parallelism | TP2 × PP2, layers `25,23`, EP ON |
| Context | `262144` |
| MTP | 2 |
| KV | FP8 E4M3 |
| Prefix cache | 기본 ON, OFF profile 포함 |
| Active concurrency | **4** |
| API | Chat Completions + `/v1/responses` |
| Multimodal | image + video |

GPU 번호와 topology는 호스트마다 다를 수 있습니다. 스크립트 기본값은 `0,1,2,3`이며 필요하면 `GPU_ORDER`로 변경할 수 있습니다.

## 설치

### 1. 모델 다운로드

원본 모델:

**https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin**

검증한 revision:

~~~text
483ee0015419568c7014b011db0c08b6ddcb2ceb
~~~

~~~bash
mkdir -p /srv/qwen38/model

hf download halt95/Qwen3.8-Flash-Next-W4A16-Merlin \
  --revision 483ee0015419568c7014b011db0c08b6ddcb2ceb \
  --local-dir /srv/qwen38/model
~~~

`model.safetensors.index.json`, 27개의 safetensors shard, `qsa_kv_scales_262k.json` 등이 필요합니다.

### 2. 저장소 clone

~~~bash
git clone https://github.com/MalloryGo/Qwen3.8-Flash-Next-4x4090-vLLM-Recipe.git
cd Qwen3.8-Flash-Next-4x4090-vLLM-Recipe
~~~

### 3. Runtime image build

일반 사용자는 patch를 직접 적용할 필요가 없습니다. Dockerfile이 공식 vLLM 0.30.0 image 위에 필요한 thin overlay를 복사합니다.

~~~bash
docker build --pull=false -t qwen38-merlin-vllm:0.30.0 runtime/
~~~

### 4. API key 생성

~~~bash
sudo mkdir -p /srv/qwen38
openssl rand -hex 32 | sudo tee /srv/qwen38/vllm-api-key >/dev/null
sudo chmod 600 /srv/qwen38/vllm-api-key
~~~

### 5. 서버 시작

~~~bash
export MODEL_DIR=/srv/qwen38/model
export CONFIG_DIR="$PWD/configs"
export API_KEY_FILE=/srv/qwen38/vllm-api-key
export RUNTIME_IMAGE=qwen38-merlin-vllm:0.30.0

bash configs/serve-262k-mtp2-prefix-on.sh
~~~

기본 bind 주소는 `127.0.0.1:8000`입니다.

GPU topology가 다르면:

~~~bash
export GPU_ORDER=0,1,2,3
~~~

Prefix cache OFF:

~~~bash
bash configs/serve-262k-mtp2-prefix-off.sh
~~~

### 6. API 확인

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

## 7개 patch

공식 vLLM 0.30.0에 다음의 작은 호환 수정만 추가합니다.

1. Qwen4Exp FP8 QSA/KV native path upstream backport
2. PP empty projected KV group 수정
3. hybrid cache PP KV-layout intersection 수정
4. PLE layer가 PP0에 있을 때 PLE + PP 허용
5. final PP stage의 MTP drafter input ownership 수정
6. Merlin `qsa_kv_scales_262k.json` 검증/적용
7. host KV-scale copy를 명시적으로 CPU에 배치

일반 사용은 `runtime/Dockerfile` build만 하면 됩니다. 상세 내용은 [docs/patches.md](docs/patches.md)를 참고하세요.

## Benchmark

| Prompt | Median TTFT | Median TPOT | Decode |
|---:|---:|---:|---:|
| 128K | 10.88 s | 4.860 ms | **205.8 tok/s** |
| 220K | 19.12 s | 4.837 ms | **206.7 tok/s** |
| 255K | 22.31 s | 5.191 ms | **192.6 tok/s** |
| 260K | 22.74 s | 4.837 ms | **206.8 tok/s** |

262,144-token combined boundary (261,888 prompt + 256 output)도 성공했습니다.

| Workload | Success | Aggregate output |
|---|---:|---:|
| 2 × 128K | 2/2 | 38.4 tok/s |
| 3 × 80K | 3/3 | 53.8 tok/s |
| 4 × 60K | 4/4 | **75.8 tok/s** |
| 100K/60K/40K/20K | 4/4 | **85.8 tok/s** |
| 12 arrivals, max active 4 | 12/12 | **95.3 tok/s** |

MTP3의 속도 이득은 작았고 MTP2가 KV capacity를 15,943 token 더 확보하여 MTP2를 권장합니다.

`/v1/responses`, streaming, tool calling, strict JSON Schema, image, video를 검증했습니다. 상세 값은 [benchmarks/results.json](benchmarks/results.json)에 있습니다.

## Known limitation

동시 **forced/required tool choice** 요청은 간헐적으로 XGrammar HTTP 500을 반환할 수 있습니다.

- 동시 Agent traffic: `tool_choice=auto`
- forced/required: 한 번에 1개 요청으로 serialize

자세한 내용: [docs/known-issues.md](docs/known-issues.md)

## License

이 저장소에서 작성한 문서, 설정, helper script는 Apache-2.0입니다. 수정된 vLLM 파일은 upstream notice와 변경 표시를 유지합니다.

모델 가중치는 재배포하지 않습니다. 원본 모델 저장소에서 받아 해당 라이선스를 준수하세요.

[NOTICE](NOTICE) / [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) / [docs/provenance.md](docs/provenance.md)