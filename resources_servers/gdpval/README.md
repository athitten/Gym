# GDPVal resources server

Scores GDPVal deliverables produced by the Stirrup agent, or by an agent such as
Pi that works in a task sandbox owned by this server (see
[Sandbox sessions](#sandbox-sessions)).

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

## Sandbox sessions

With `sandbox_provider` set (see `configs/gdpval_pi_sandbox.yaml`), the server
starts one sandbox per task, stages the task's reference files in
`/workspace/input` and lends the sandbox to the agent. At verify time it copies
the agent's files from `/workspace/output` into a new
`<deliverables_root>/gdp-*/` directory and judges that copy:

- Only regular files directly in `/workspace/output` are exported.
  Subdirectories, links and unsafe file names are skipped.
- Run-state file names (`finish_params.json`, `history.json`, `metadata.json`,
  `log.txt` and the others in `nemo_gym/deliverables.py`) are never exported.
- At most 100 files and 1 GiB are exported per attempt; files past either limit
  are skipped.
- The export's `finish_params.json` lists the exported `paths` and the
  `skipped` entries with a reason (the first 100). `skipped_count` has the
  total, including any entries past the first 10,000, which are never listed.
- A missing or symlinked `/workspace/output` is recorded in `skipped` too, so it
  can be told apart from an empty one. Both are judged as an empty submission.
- An invalid judge verdict, or an error raised while judging, keeps its export
  and is returned with `mask_sample: true`, so its placeholder reward is not
  counted.

To score these exports with multi-stage ELO, or to judge them again later, copy
them into a judge-only cache with `cache_sandbox_deliverables.py`; see "Fresh vs.
cached deliverables" in `benchmarks/gdpval/README.md`.
