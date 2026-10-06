# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Codex x GDPVal sandbox pairing: composition only, with each side's real session code."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

import resources_servers.gdpval.app as gdp_app
from nemo_gym.base_resources_server import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.global_config import ALLOW_UNSUPPORTED_PAIRING_ENV_VAR_NAME, GlobalConfigDictParser, allowed_agents_for
from nemo_gym.rollout_collection import NG_ENVIRONMENT_SERVER_KEY, RolloutCollectionHelper
from nemo_gym.server_utils import ServerClient
from resources_servers.gdpval.app import (
    GDPValResourcesServer,
    GDPValResourcesServerConfig,
    GDPValVerifyRequest,
    GDPValVerifyResponse,
)
from resources_servers.gdpval.task_data import INPUT_DIR, OUTPUT_DIR, WORKDIR, prepare_row
from responses_api_agents.codex_agent.app import CodexAgent, CodexAgentConfig

# Codex's own fake task sandbox: it plays the in-sandbox supervisor protocol the real Codex session expects.
from responses_api_agents.codex_agent.tests.test_native_sessions import Sandbox as CodexTaskSandbox


CONFIG_PATHS = [
    "resources_servers/gdpval/configs/gdpval_sandbox.yaml",
    "responses_api_agents/codex_agent/configs/codex_agent.yaml",
    "environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml",
]
# The README composition: only the Environment Server binding names the pair.
BINDING = {
    "policy_model_name": "test-policy",
    "single_agent_turn_legacy": {
        "environment_servers": {
            "single_agent_turn_legacy": {
                "resources_server": {"name": "gdpval_resources_server"},
                "agent_server": {"name": "codex_agent"},
                "resources_tool_transports": [],
            }
        }
    },
}


def compose():
    _, layers = GlobalConfigDictParser().load_extra_config_paths(CONFIG_PATHS)
    return OmegaConf.merge(*layers, BINDING)


@pytest.mark.parametrize("declared", [None, ["pi_agent"]])
def test_gdp_sandbox_declares_codex_for_environment_server_rows(declared, monkeypatch):
    monkeypatch.delenv(ALLOW_UNSUPPORTED_PAIRING_ENV_VAR_NAME, raising=False)
    config = compose()
    assert allowed_agents_for(config, "gdpval_resources_server") == ["codex_agent"]
    assert "pi_agent" not in config and "stirrup_agent" not in config  # The benchmark config brings no agent.
    if declared is not None:
        OmegaConf.update(config, "gdpval_resources_server.resources_servers.gdpval.allowed_agents", declared)
    # Rows that name the Environment Server are checked against the benchmark's allowed_agents before dispatch.
    rows = [{NG_ENVIRONMENT_SERVER_KEY: "single_agent_turn_legacy"}]
    if declared is None:
        RolloutCollectionHelper._validate_environment_servers(rows, config)
    else:
        with pytest.raises(ValueError, match="accepts only: pi_agent"):
            RolloutCollectionHelper._validate_environment_servers(rows, config)


class TaskSandbox(CodexTaskSandbox):
    """One sandbox for both sides: GDPVal creates and lists it, Codex borrows it and writes a deliverable."""

    def __init__(self):
        super().__init__()
        self.expected_workdir = WORKDIR
        self.start = AsyncMock()
        self.serialize = AsyncMock(return_value={"sandbox_id": "task-box"})

    async def upload(self, source, destination):
        self.files[destination] = Path(source).read_bytes()

    async def download(self, source, destination):
        data = self.files[source]
        Path(destination).write_bytes(data if isinstance(data, bytes) else data.encode())

    async def execute(self, command, **kwargs):
        if command.startswith("python3 -c ") and repr(OUTPUT_DIR) in command:  # GDPVal's export listing
            outputs = [
                {"name": path.removeprefix(f"{OUTPUT_DIR}/"), "size": len(data), "regular": True}
                for path, data in self.files.items()
                if path.startswith(f"{OUTPUT_DIR}/")
            ]
            return SimpleNamespace(return_code=0, stdout=json.dumps(outputs), stderr="", error_type=None)
        return await super().execute(command, **kwargs)

    async def create(self, **kwargs):
        self.files[f"{OUTPUT_DIR}/summary.md"] = b"# audit summary\n"
        # Bound the fake's only wait so a regression fails instead of hanging the suite.
        return await asyncio.wait_for(super().create(**kwargs), timeout=30)


async def test_codex_deliverable_from_composed_pair_is_graded_after_close(tmp_path, monkeypatch):
    config = compose()
    gdp = config.gdpval_resources_server.resources_servers.gdpval
    gdp.image, gdp.deliverables_root = "gdp.sif", str(tmp_path)
    assert config.sandbox.apptainer.create.mount_point == WORKDIR
    row = prepare_row(json.loads((Path(__file__).parents[1] / "data/example.jsonl").read_text().splitlines()[0]))

    box = TaskSandbox()
    providers = []
    monkeypatch.setattr(gdp_app, "AsyncSandbox", lambda provider: providers.append(provider) or box)
    monkeypatch.setattr(gdp_app, "get_global_config_dict", lambda: config)

    async def chunks(*args):
        yield b"PK reference"

    download = SimpleNamespace(
        raise_for_status=MagicMock(), release=MagicMock(), content=SimpleNamespace(iter_chunked=chunks)
    )
    monkeypatch.setattr(gdp_app, "http_request", AsyncMock(return_value=download))
    graded = []

    async def grade(self, body):
        graded.append(sorted(path.name for path in Path(body.deliverables_dir).iterdir()))
        return GDPValVerifyResponse(**body.model_dump(), reward=1.0)

    monkeypatch.setattr(GDPValResourcesServer, "_grade_deliverables", grade)
    resources = GDPValResourcesServer(
        config=GDPValResourcesServerConfig.model_validate(
            OmegaConf.to_container(gdp, resolve=True)
            | {"name": "gdpval_resources_server", "host": "h", "port": 1, "preconvert_office_to_pdf": False}
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    episode = EpisodeId(rollout_id="gdp-codex")
    task_id = TaskId(taskset="gdpval_resources_server", task_id=row["task_id"])
    request = SimpleNamespace(session={})
    seed = ResourcesSeedSessionRequest(resources_session_id="r-1", episode_id=episode, task_id=task_id, task_data=row)
    access = (await resources.seed_session(request, seed)).sandbox_access
    assert providers == [OmegaConf.to_container(config.sandbox)]
    assert box.files[f"{INPUT_DIR}/{row['reference_files'][0]}"] == b"PK reference"

    # The stock harness config, unchanged apart from the server address.
    agent_config = CodexAgentConfig.model_validate(
        OmegaConf.to_container(config.codex_agent.responses_api_agents.codex_agent, resolve=True)
        | {"name": "codex_agent", "host": "localhost", "port": 8001}
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = OmegaConf.create({"policy_model": {"responses_api_models": {"openai_model": {}}}})
    client._build_server_base_url.return_value = "http://model.example:9000"
    module = "responses_api_agents.codex_agent.app"
    with (
        patch(f"{module}.ensure_codex", side_effect=AssertionError("native sessions must not install host Codex")),
        patch(f"{module}.get_global_config_dict", return_value=config),
        patch(f"{module}.create_provider") as create_provider,
        patch(f"{module}.AsyncSandbox.connect", AsyncMock(return_value=box)) as connect,
    ):
        agent = CodexAgent(config=agent_config, server_client=client)
        with TestClient(agent.setup_webserver()) as http:
            body = AgentSeedSessionRequest(
                agent_session_id="a-1", episode_id=episode, task_id=task_id, sandbox_access=access
            )
            created = http.post("/v1/agent_sessions", json=body.model_dump(mode="json"))
            assert created.status_code == 200, created.text
            result = http.post(f"/ng-rollout/{episode.capture_key}/v1/responses", json=row["responses_create_params"])
            assert result.status_code == 200, result.text
            payload = json.loads(box.files[f"{box.directory}/input.json"])
            closed = http.post(
                "/v1/agent_sessions/close", json={"agent_session_id": "a-1", "episode_id": episode.model_dump()}
            )
            assert closed.status_code == 200, closed.text

    # Codex connected to the sandbox GDPVal lent, through the same provider block, and ran the task prompt there.
    assert create_provider.call_args.args[0] == OmegaConf.to_container(config.sandbox)
    assert connect.await_args.args[0] == {"sandbox_id": "task-box"}
    assert payload["cwd"] == WORKDIR
    assert payload["prompt"] == row["responses_create_params"]["input"][0]["content"]
    box.disconnect.assert_awaited_once()
    box.stop.assert_not_awaited()  # The borrower never destroys the task sandbox.

    verdict = await resources.verify(GDPValVerifyRequest(**row, response=result.json()), request=request)
    assert verdict.reward == 1.0
    assert graded == [["finish_params.json", "summary.md"]]
    box.stop.assert_not_awaited()
    close = ResourcesCloseSessionRequest(resources_session_id="r-1", episode_id=episode)
    await resources.close_resources_session(request, close)
    box.stop.assert_awaited_once()
