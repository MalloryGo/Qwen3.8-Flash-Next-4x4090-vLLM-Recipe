# Known issues and operational notes

## Concurrent forced/required tool choice

The main functional limitation observed with this runtime is constrained tool
choice under concurrency:

- concurrent `tool_choice=auto`: tested successfully;
- single forced/required tool choice: tested successfully;
- concurrent forced/required tool choice: intermittent XGrammar HTTP 500.

Recommended policy: use `tool_choice=auto` for concurrent Agent traffic and
serialize forced/required requests to one in-flight request.

## Prefix-cache scope

Repeated-prefix and mixed-prefix probes with 2–4 concurrent requests completed
without NaN, constant-token loops or prompt-independent output.

The upstream hybrid Mamba/prefix-cache issue tracked as `#55506` was not
reproduced in those bounded tests. This is not a universal correctness guarantee.
If a workload exposes a problem, use the supplied prefix-cache-OFF profile.

## Responses store retention

`VLLM_ENABLE_RESPONSES_API_STORE=1` is enabled so Responses retrieve/cancel
flows work. Treat response retention as an operational concern for long-running
services; a gateway or serving layer should define lifecycle policy.

## VRAM headroom

The validated workload runs close to the 24 GB/GPU limit. Keep active inference
concurrency at **4** and queue larger bursts. Near-capacity workloads should be
monitored for allocator pressure.

## FP8 numerical tolerance

A strict numerical reference suite recorded 84 passes and 5 assertions outside
`atol=1e-5`; the largest selected-logit absolute difference was about
`1.0e-4`. Long-context serving, API correctness and multimodal checks completed
successfully. This is documented as a precision caveat.
