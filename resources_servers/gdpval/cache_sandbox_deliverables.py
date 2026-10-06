# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copy sandbox-session exports into the layout that judge-only GDPVal scoring reads.

Multi-stage ELO runs through the GDPVal benchmark's rollout driver, which ``gym eval run --no-serve``
does not use. Sandbox-session harnesses such as Pi are therefore scored in two passes: collect the
rollouts once, build this cache, then run judge-only multi-stage ELO with ``PERSIST_DELIVERABLES_DIR``
pointing at it. Each cached attempt is ``<output>/task_<task_id>/repeat_<rollout_index>/``. Copies are staged
in ``<output>/.staging/``, which is removed at the end.

    python -m resources_servers.gdpval.cache_sandbox_deliverables \
        --rollouts rollouts.jsonl --output /abs/path/deliverables-cache
"""

import argparse
import json
import shutil
from pathlib import Path


MARKER = "finish_params.json"
STAGING = ".staging"


def _copy_attempt(source: Path, target: Path, staging_root: Path) -> None:
    """Copy *source* to *target* so that *target* appears only once the copy is complete.

    The copy is staged outside every ``task_<id>`` directory: judge-only scoring treats any ``repeat_*``
    directory there as an attempt, so a copy left behind by a failure must never sit beside the repeats.
    """
    staging = staging_root / f"{target.parent.name}.{target.name}"
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(source, staging)
    target.parent.mkdir(exist_ok=True)
    staging.rename(target)


def build_cache(rollouts: Path, output: Path) -> dict[str, int]:
    """Copy every completed export listed in *rollouts* into *output*; return copy counts."""
    output.mkdir(parents=True, exist_ok=True)
    staging_root = output / STAGING
    counts = {"cached": 0, "skipped_without_export": 0}
    seen: set[tuple[str, int]] = set()
    try:
        with rollouts.open() as handle:
            for line in handle:
                row = json.loads(line)
                task_id, source = row.get("task_id"), row.get("deliverables_dir")
                if not isinstance(task_id, str) or not source or not (Path(source) / MARKER).is_file():
                    counts["skipped_without_export"] += 1
                    continue
                index = int(row.get("_ng_rollout_index") or 0)
                if (task_id, index) in seen:
                    raise ValueError(f"Duplicate rollout for task {task_id} repeat {index}")
                seen.add((task_id, index))
                target = output / f"task_{task_id}" / f"repeat_{index}"
                if target.exists():
                    raise FileExistsError(f"{target} already exists; use a fresh output directory")
                _copy_attempt(Path(source), target, staging_root)
                counts["cached"] += 1
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollouts", type=Path, required=True, help="rollouts.jsonl from the sandbox-session run")
    parser.add_argument("--output", type=Path, required=True, help="absolute cache directory to create")
    args = parser.parse_args(argv)
    print(json.dumps(build_cache(args.rollouts, args.output)))


if __name__ == "__main__":
    main()
