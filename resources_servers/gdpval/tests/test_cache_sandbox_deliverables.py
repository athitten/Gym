# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from resources_servers.gdpval.app import _iter_ref_repeat_dirs
from resources_servers.gdpval.cache_sandbox_deliverables import build_cache, main
from resources_servers.gdpval.comparison import task_attempted


def _export(root, name, files, marker=True):
    directory = root / name
    directory.mkdir(parents=True)
    for file_name, content in files.items():
        (directory / file_name).write_text(content)
    if marker:
        (directory / "finish_params.json").write_text(json.dumps({"paths": sorted(files)}))
    return directory


def _rollouts(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_cache_matches_judge_only_layout(tmp_path):
    first = _export(tmp_path / "exports", "gdp-a", {"report.xlsx": "a"})
    second = _export(tmp_path / "exports", "gdp-b", {})
    rollouts = _rollouts(
        tmp_path / "rollouts.jsonl",
        [
            {"task_id": "t1", "_ng_rollout_index": 0, "deliverables_dir": str(first), "reward": 0.5},
            {"task_id": "t2", "_ng_rollout_index": 1, "deliverables_dir": str(second), "reward": 0.0},
        ],
    )
    counts = build_cache(rollouts, tmp_path / "cache")
    assert counts == {"cached": 2, "skipped_without_export": 0}
    cached = tmp_path / "cache" / "task_t1" / "repeat_0"
    assert sorted(p.name for p in cached.iterdir()) == ["finish_params.json", "report.xlsx"]
    # An attempt that exported nothing is still finished, so it is judged rather than reported missing.
    assert task_attempted(str(tmp_path / "cache" / "task_t2" / "repeat_1"))
    assert [p.name for p in _iter_ref_repeat_dirs(tmp_path / "cache" / "task_t1")] == ["repeat_0"]
    assert not list((tmp_path / "cache").rglob("*.partial"))


def test_rows_without_completed_export_are_skipped(tmp_path):
    unfinished = _export(tmp_path / "exports", "gdp-a", {"report.xlsx": "a"}, marker=False)
    rollouts = _rollouts(
        tmp_path / "rollouts.jsonl",
        [
            {"task_id": "t1", "deliverables_dir": str(unfinished)},
            {"task_id": "t2"},
            {"deliverables_dir": str(unfinished)},
        ],
    )
    assert build_cache(rollouts, tmp_path / "cache") == {"cached": 0, "skipped_without_export": 3}
    assert not (tmp_path / "cache" / "task_t1").exists()


def test_duplicate_attempt_rejected(tmp_path):
    export = _export(tmp_path / "exports", "gdp-a", {"report.xlsx": "a"})
    row = {"task_id": "t1", "_ng_rollout_index": 0, "deliverables_dir": str(export)}
    with pytest.raises(ValueError, match="Duplicate"):
        build_cache(_rollouts(tmp_path / "rollouts.jsonl", [row, row]), tmp_path / "cache")


def test_existing_cache_entry_is_never_overwritten(tmp_path, capsys):
    export = _export(tmp_path / "exports", "gdp-a", {"report.xlsx": "new"})
    existing = tmp_path / "cache" / "task_t1" / "repeat_0"
    existing.mkdir(parents=True)
    (existing / "report.xlsx").write_text("old")
    rollouts = _rollouts(tmp_path / "rollouts.jsonl", [{"task_id": "t1", "deliverables_dir": str(export)}])
    with pytest.raises(FileExistsError):
        main(["--rollouts", str(rollouts), "--output", str(tmp_path / "cache")])
    assert (existing / "report.xlsx").read_text() == "old"
