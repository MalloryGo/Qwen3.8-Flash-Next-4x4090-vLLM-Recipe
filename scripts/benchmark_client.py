#!/usr/bin/env python3
"""Small stdlib-only streaming benchmark client for this Merlin recipe."""

from __future__ import annotations

import argparse
import os
import concurrent.futures
import hashlib
import json
import pathlib
import statistics
import subprocess
import time
import urllib.error
import urllib.request


METRIC_NAMES = (
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_decode_time_seconds_sum",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_per_pos_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
)


def fetch_metrics(url: str) -> dict[tuple[str, str], float]:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=10) as resp:
            text = resp.read().decode("utf-8", "replace")
    except Exception:
        return {}
    out: dict[tuple[str, str], float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        metric, raw_value = parts[0], parts[1]
        name, _, labels = metric.partition("{")
        labels = labels.rstrip("}")
        if not any(name == n or name.startswith(n + "{") for n in METRIC_NAMES):
            if not any(name == n for n in METRIC_NAMES):
                continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if value == value and abs(value) != float("inf"):
            out[(name, labels)] = value
    return out


def metric_sum(snapshot: dict[tuple[str, str], float], name: str) -> float:
    return sum(value for (metric_name, _), value in snapshot.items() if metric_name == name)


def metric_delta(
    before: dict[tuple[str, str], float],
    after: dict[tuple[str, str], float],
    name: str,
) -> float:
    keys = {key for key in before if key[0] == name} | {
        key for key in after if key[0] == name
    }
    return sum(after.get(key, 0.0) - before.get(key, 0.0) for key in keys)


def metric_delta_by_position(
    before: dict[tuple[str, str], float], after: dict[tuple[str, str], float]
) -> dict[str, int]:
    name = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
    keys = {key for key in before if key[0] == name} | {
        key for key in after if key[0] == name
    }
    result: dict[str, int] = {}
    for key in keys:
        labels = key[1]
        pos = "unknown"
        for item in labels.split(","):
            if item.startswith('position="'):
                pos = item.split('"')[1]
        delta = after.get(key, 0.0) - before.get(key, 0.0)
        if delta:
            result[pos] = int(delta)
    return result


def summarize_metrics(
    before: dict[tuple[str, str], float], after: dict[tuple[str, str], float]
) -> dict[str, object]:
    names = [
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:prompt_tokens_cached_total",
        "vllm:request_prefill_time_seconds_sum",
        "vllm:request_decode_time_seconds_sum",
        "vllm:time_to_first_token_seconds_sum",
        "vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
    ]
    delta = {name: metric_delta(before, after, name) for name in names}
    drafts = delta["vllm:spec_decode_num_drafts_total"]
    draft_tokens = delta["vllm:spec_decode_num_draft_tokens_total"]
    accepted = delta["vllm:spec_decode_num_accepted_tokens_total"]
    delta["spec_decode_acceptance_rate"] = accepted / draft_tokens if draft_tokens else None
    delta["spec_decode_mean_accepted_length"] = 1 + accepted / drafts if drafts else None
    delta["spec_decode_accepted_by_position"] = metric_delta_by_position(before, after)
    return delta


def memory_snapshot() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "MemFree", "Mlocked", "SwapTotal", "SwapFree"}:
                values[key + "_kib"] = int(rest.strip().split()[0])
    except OSError:
        pass
    return values


def start_gpu_monitor(path: pathlib.Path):
    fp = path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
            "-l",
            "1",
        ],
        stdout=fp,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, fp


def stop_gpu_monitor(proc, fp) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    fp.close()


def run_request(
    *,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    seed: int,
    timeout: int,
    capture_metrics: bool,
) -> dict[str, object]:
    before = fetch_metrics(url) if capture_metrics else {}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **({"Authorization": "Bearer " + os.environ["API_KEY"]} if os.environ.get("API_KEY") else {})},
        method="POST",
    )
    started = time.monotonic()
    first_content_s = None
    content_parts: list[str] = []
    usage = None
    finish_reason = None
    http_status = None
    error = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            http_status = resp.status
            for raw_line in resp:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    piece = delta.get("content") or delta.get("reasoning") or ""
                    if piece:
                        if first_content_s is None:
                            first_content_s = time.monotonic() - started
                        content_parts.append(piece)
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
    except urllib.error.HTTPError as exc:
        http_status = exc.code
        error = "HTTP request failed"
    except Exception as exc:
        error = type(exc).__name__
    finished = time.monotonic()
    if capture_metrics:
        time.sleep(0.35)
        after = fetch_metrics(url)
        metric_info = summarize_metrics(before, after)
    else:
        metric_info = None
    elapsed = finished - started
    content = "".join(content_parts)
    prompt_tokens = usage.get("prompt_tokens") if usage else None
    completion_tokens = usage.get("completion_tokens") if usage else None
    if metric_info:
        if prompt_tokens is None:
            prompt_tokens = int(metric_info.get("vllm:prompt_tokens_total", 0)) or None
        if completion_tokens is None:
            completion_tokens = int(metric_info.get("vllm:generation_tokens_total", 0)) or None
    server_prefill = (metric_info or {}).get("vllm:request_prefill_time_seconds_sum", 0.0)
    server_decode = (metric_info or {}).get("vllm:request_decode_time_seconds_sum", 0.0)
    server_ttft = (metric_info or {}).get("vllm:time_to_first_token_seconds_sum", 0.0)
    ttft = server_ttft or first_content_s
    result: dict[str, object] = {
        "http_status": http_status,
        "error": error,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "client_elapsed_s": elapsed,
        "client_ttft_s": first_content_s,
        "server_ttft_s": server_ttft or None,
        "server_prefill_s": server_prefill or None,
        "server_decode_s": server_decode or None,
        "prefill_tok_s": (prompt_tokens / server_prefill) if prompt_tokens and server_prefill else (prompt_tokens / ttft if prompt_tokens and ttft else None),
        "decode_tok_s": (completion_tokens / server_decode) if completion_tokens and server_decode else ((completion_tokens - 1) / max(elapsed - (first_content_s or 0.0), 1e-9) if completion_tokens else None),
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "content_chars": len(content),
        "metrics_delta": metric_info,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.8-flash")
    parser.add_argument("--prompt-files", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-stem", required=True)
    parser.add_argument("--max-tokens", type=int, default=1280)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--warmup-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = [pathlib.Path(p).read_text(encoding="utf-8") for p in args.prompt_files]
    prompt_files = [pathlib.Path(p) for p in args.prompt_files]
    gpu_path = out_dir / f"{args.output_stem}.gpu.csv"
    monitor_proc, monitor_fp = start_gpu_monitor(gpu_path)
    result: dict[str, object] = {
        "output_stem": args.output_stem,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "warmups": args.warmups,
        "warmup_tokens": args.warmup_tokens,
        "repeats": args.repeats,
        "concurrency": args.concurrency,
        "seed": args.seed,
        "chat_template_kwargs": {"enable_thinking": False},
        "prompt_files": [
            {
                "input_index": prompt_files.index(path),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in prompt_files
        ],
        "memory_before_kib": memory_snapshot(),
        "warmup_results": [],
        "measured_results": [],
    }
    try:
        for index in range(args.warmups):
            warm = run_request(
                url=args.url,
                model=args.model,
                prompt=prompts[index % len(prompts)],
                max_tokens=args.warmup_tokens,
                seed=args.seed,
                timeout=args.timeout,
                capture_metrics=False,
            )
            warm["index"] = index + 1
            result["warmup_results"].append(warm)

        if args.concurrency == 1:
            for index in range(args.repeats):
                trial = run_request(
                    url=args.url,
                    model=args.model,
                    prompt=prompts[index % len(prompts)],
                    max_tokens=args.max_tokens,
                    seed=args.seed,
                    timeout=args.timeout,
                    capture_metrics=True,
                )
                trial["index"] = index + 1
                result["measured_results"].append(trial)

        else:
            for wave in range(args.repeats):
                before = fetch_metrics(args.url)
                wave_prompts = [prompts[i % len(prompts)] for i in range(args.concurrency)]
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    futures = [
                        pool.submit(
                            run_request,
                            url=args.url,
                            model=args.model,
                            prompt=prompt,
                            max_tokens=args.max_tokens,
                            seed=args.seed + wave * args.concurrency + i,
                            timeout=args.timeout,
                            capture_metrics=False,
                        )
                        for i, prompt in enumerate(wave_prompts)
                    ]
                    batch = [future.result() for future in futures]
                time.sleep(0.35)
                after = fetch_metrics(args.url)
                group_metrics = summarize_metrics(before, after)
                result["measured_results"].append(
                    {"wave": wave + 1, "group_metrics_delta": group_metrics, "requests": batch}
                )

    finally:
        result["memory_after_kib"] = memory_snapshot()
        stop_gpu_monitor(monitor_proc, monitor_fp)

    if args.concurrency == 1:
        keys = ("prompt_tokens", "client_ttft_s", "server_ttft_s", "server_prefill_s", "server_decode_s", "prefill_tok_s", "decode_tok_s")
        summary = {}
        trials = result["measured_results"]
        for key in keys:
            values = [x[key] for x in trials if isinstance(x.get(key), (int, float))]
            summary[key] = {"median": statistics.median(values), "values": values} if values else None
        result["summary"] = summary
    result_path = out_dir / f"{args.output_stem}.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"result": str(result_path), "summary": result.get("summary"), "errors": [x.get("error") for x in result["measured_results"] if isinstance(x, dict) and x.get("error")]}, ensure_ascii=False, indent=2))
    return 0 if all(not x.get("error") and x.get("http_status") == 200 for x in result["warmup_results"] + (result["measured_results"] if args.concurrency == 1 else [])) else 1


if __name__ == "__main__":
    raise SystemExit(main())