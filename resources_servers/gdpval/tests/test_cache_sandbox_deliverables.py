# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import shutil

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
    outcomes = build_cache(rollouts, tmp_path / "cache")
    assert outcomes == {"cached": ["t1", "t2"], "skipped_without_export": [], "export_without_marker": []}
    cached = tmp_path / "cache" / "task_t1" / "repeat_0"
    assert sorted(p.name for p in cached.iterdir()) == ["finish_params.json", "report.xlsx"]
    # An attempt that exported nothing is still finished, so it is judged rather than reported missing.
    assert task_attempted(str(tmp_path / "cache" / "task_t2" / "repeat_1"))
    assert [p.name for p in _iter_ref_repeat_dirs(tmp_path / "cache" / "task_t1")] == ["repeat_0"]
    # The staging area is removed once the cache is built.
    assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == ["task_t1", "task_t2"]


def test_failed_copy_never_looks_like_an_attempt(tmp_path, monkeypatch):
    export = _export(tmp_path / "exports", "gdp-a", {"a.xlsx": "a", "b.xlsx": "b"})
    rollouts = _rollouts(tmp_path / "rollouts.jsonl", [{"task_id": "t1", "deliverables_dir": str(export)}])
    copytree = shutil.copytree
    copied = []

    def copy_one_then_fail(source, destination):
        if copied:
            raise OSError("disk full")
        copied.append(destination)
        return shutil.copy2(source, destination)

    monkeypatch.setattr(
        shutil, "copytree", lambda source, target: copytree(source, target, copy_function=copy_one_then_fail)
    )
    with pytest.raises(OSError):
        build_cache(rollouts, tmp_path / "cache")
    assert copied
    # Judge-only scoring would treat a leftover repeat_* directory under task_t1 as an attempt.
    assert _iter_ref_repeat_dirs(tmp_path / "cache" / "task_t1") == []
    assert list((tmp_path / "cache").iterdir()) == []


def test_rows_without_an_export_are_a_plain_skip(tmp_path, capsys):
    finished = _export(tmp_path / "exports", "gdp-a", {"report.xlsx": "a"})
    rollouts = _rollouts(
        tmp_path / "rollouts.jsonl",
        [{"task_id": "t1", "deliverables_dir": str(finished)}, {"task_id": "t2"}, {"task_id": "t3"}],
    )
    main(["--rollouts", str(rollouts), "--output", str(tmp_path / "cache")])
    assert json.loads(capsys.readouterr().out) == {
        "cached": {"count": 1, "task_ids": ["t1"]},
        "skipped_without_export": {"count": 2, "task_ids": ["t2", "t3"]},
        "export_without_marker": {"count": 0, "task_ids": []},
    }


def test_export_without_marker_fails_the_build(tmp_path, capsys):
    finished = _export(tmp_path / "exports", "gdp-a", {"report.xlsx": "a"})
    unfinished = _export(tmp_path / "exports", "gdp-b", {"report.xlsx": "b"}, marker=False)
    rollouts = _rollouts(
        tmp_path / "rollouts.jsonl",
        [
            {"task_id": "t1", "deliverables_dir": str(finished)},
            {"task_id": "t2", "deliverables_dir": str(unfinished)},
            {"task_id": "t3", "deliverables_dir": str(tmp_path / "exports" / "moved")},
            {"task_id": "t4"},
        ],
    )
    with pytest.raises(SystemExit) as raised:
        main(["--rollouts", str(rollouts), "--output", str(tmp_path / "cache")])
    assert "['t2', 't3']" in str(raised.value.code)
    assert json.loads(capsys.readouterr().out) == {
        "cached": {"count": 1, "task_ids": ["t1"]},
        "skipped_without_export": {"count": 1, "task_ids": ["t4"]},
        "export_without_marker": {"count": 2, "task_ids": ["t2", "t3"]},
    }
    # Finished exports are still cached; an export without its marker never is.
    assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == ["task_t1"]


def test_finished_export_without_task_id_is_rejected(tmp_path):
    export = _export(tmp_path / "exports", "gdp-a", {"report.xlsx": "a"})
    rollouts = _rollouts(tmp_path / "rollouts.jsonl", [{"deliverables_dir": str(export)}])
    with pytest.raises(ValueError, match="task_id"):
        build_cache(rollouts, tmp_path / "cache")


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
