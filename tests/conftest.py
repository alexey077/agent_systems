import uuid
from pathlib import Path

import pytest

from citizen_agent.config import load_config
from citizen_agent.models import Session
from citizen_agent.storage import Store
from citizen_agent.tools import Executor
from citizen_agent.trace import Trace

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path):
    cfg, _ = load_config(ROOT / "config/offline.toml")
    cfg.database_path = str(tmp_path / "appeals.sqlite3")
    cfg.traces_path = str(tmp_path / "traces")
    return cfg


@pytest.fixture
def session():
    return Session(request_id="test_case", actor_id="test_operator")


@pytest.fixture
def executor_factory(config, session):
    def factory(**kwargs):
        trace = Trace(config.traces_path, "test_" + uuid.uuid4().hex)
        return Executor(
            knowledge_path=ROOT / config.knowledge_path,
            routing_path=ROOT / config.routing_path,
            store=Store(config.database_path),
            session=kwargs.pop("session", session),
            runtime=config.runtime,
            trace=trace,
            **kwargs,
        )

    return factory
