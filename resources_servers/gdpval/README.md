# GDPVal resources server

Scores deliverables produced by the Stirrup agent on the GDPVal benchmark.

Two modes via `reward_mode` config:

- `rubric` (default) — LLM judge scores each deliverable against a per-task
  rubric, reward in `[0.0, 1.0]`.
- `comparison` — pairwise judge compares eval deliverable vs. one or more
  reference rollouts (`reference_deliverables_dir`, or `reference_models` for
  multi-reference), reward in `{0.0, 0.5, 1.0}`. `aggregate_metrics` reduces to
  an ELO rating.

Comparison mode also supports **multi-stage adaptive ELO** — a sequence of
stages that judge sampled tasks against an adaptively-chosen reference subset,
enabled with `++multistage.enabled=true`. It is implemented in
`multistage_orchestrator.py` (pure logic in `multistage_elo.py`) and runs through
the standard `gym eval run` pipeline. See the "Run multi-stage adaptive ELO"
section of `benchmarks/gdpval/README.md`.

Canonical entry point is the benchmark at `benchmarks/gdpval/`:

```bash
gym eval prepare --benchmark gdpval
gym eval run \
  --model-type vllm_model \
  --benchmark gdpval \
  --split benchmark
```

See `benchmarks/gdpval/README.md` for the full run recipe.

## Sandboxed harness sessions (experimental)

With `sandbox_provider` set, the server owns one task sandbox per episode and lends it to a harness that runs
inside it. Stirrup and judge-only runs are unchanged.

1. Seed creates the sandbox from `image` and stages reference files under `/workspace/input`.
2. The harness borrows the sandbox and writes deliverables to `/workspace/output`.
3. After the agent closes, verify copies those files to a new directory under `deliverables_root`, writes
   `finish_params.json`, and grades them. Closing the resources session destroys the sandbox.

`configs/gdpval_sandbox.yaml` is the benchmark side only. Compose it with an agent listed in its `allowed_agents`
and `single_agent_turn_legacy`. For Codex, save this as `run.yaml`:

```yaml
config_paths:
  - resources_servers/gdpval/configs/gdpval_sandbox.yaml
  - responses_api_agents/codex_agent/configs/codex_agent.yaml
  - environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml

gdpval_resources_server:
  resources_servers:
    gdpval:
      image: /absolute/path/gdpval.sif
      deliverables_root: /absolute/path/deliverables

single_agent_turn_legacy:
  environment_servers:
    single_agent_turn_legacy:
      resources_server:
        name: gdpval_resources_server
      agent_server:
        name: codex_agent
      resources_tool_transports: []
```

Supply `policy_model`, `policy_model_name`, and `gdpval_judge_model` in `model-provider.yaml`. Codex sends no
sampling or output-limit fields, so set them on the model server. With the `openai_model` adapter, put
`temperature`, `top_p`, and `max_output_tokens` in its `extra_body`. Set the Codex agent's `timeout` (600 s by
default; work cut off by it is still graded) and `model_context_window` / `model_auto_compact_token_limit` for the
served model; see the Codex agent README.

```bash
gym eval prepare --benchmark gdpval
python -m resources_servers.gdpval.task_data \
  --input benchmarks/gdpval/data/gdpval_benchmark.jsonl --output prepared.jsonl

gym env start --config run.yaml --config model-provider.yaml

gym eval run --no-serve --config run.yaml --config model-provider.yaml \
  --agent codex_agent -i prepared.jsonl -o rollouts.jsonl --concurrency 4
```

`task_data` refuses to overwrite an existing output file. `--concurrency` sets how many episodes run at once. The
Environment Server's `max_concurrent_episodes` is unlimited by default; if you set it, keep it at or above
`--concurrency` and also set `queue_timeout_seconds`.

- Use the audited GDPVal image built from `responses_api_agents/stirrup_agent/containers/gdpval.def`. The harness
  sees only the task prompt and reference files; the rubric stays on the server.
- `gdpval_sandbox.yaml` defines the top-level `sandbox` provider block that `sandbox_provider` names. Don't define
  another provider under `sandbox` in your own configs: the merged block must hold exactly one provider.
- Codex installs its pinned runtime into each sandbox from nodejs.org and the npm registry, so the sandbox needs
  outbound network access.
- Where Apptainer cannot join an instance with `--fakeroot` (no `/etc/subuid` entry), start instances with
  `sandbox.apptainer.create.extra_start_args: [--fakeroot, --containall, --writable-tmpfs]` and set
  `sandbox.apptainer.exec.fakeroot_for_root: false`.
- Without LibreOffice on the host, set `gdpval_resources_server.resources_servers.gdpval.libreoffice_command` to run
  the image's LibreOffice, for example `[apptainer, exec, --bind, /data, --bind, /tmp, /absolute/path/gdpval.sif,
  libreoffice]`. It must see the deliverable, reference, and temporary directories at the same paths.
- Converted Office files reach the judge as PDFs. A judge that rejects PDF data URLs (for example GPT-5.5 behind
  an Azure route, which returns HTTP 400) needs `judge_media_mode: images_and_text`.
- `gym eval run --no-serve` does not use `rollout_collection_driver`, so multi-stage ELO does not run on this path.
  To score with ELO, copy the exports into a judge-only cache with
  `python -m resources_servers.gdpval.cache_sandbox_deliverables --rollouts rollouts.jsonl --output /absolute/cache`
  and run the benchmark's judge-only comparison with `PERSIST_DELIVERABLES_DIR` set to it
  (see `benchmarks/gdpval/README.md`).
