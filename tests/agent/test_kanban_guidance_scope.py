"""Board tools alone must not turn an ordinary chat into a worker."""
from types import SimpleNamespace

import pytest
from agent.system_prompt import _tool_guidance_block


@pytest.mark.parametrize('task,expected', [(None, False), ('task-example', True)])
def test_board_guidance_requires_task(monkeypatch, task, expected):
    monkeypatch.delenv('HERMES_KANBAN_TASK', raising=False)
    if task:
        monkeypatch.setenv('HERMES_KANBAN_TASK', task)
    agent = SimpleNamespace(valid_tool_names={'kanban_show'})
    assert ('Kanban task execution protocol' in (_tool_guidance_block(agent) or '')) is expected


def test_explicit_empty_snapshot_stays_empty(monkeypatch):
    agent = SimpleNamespace(valid_tool_names={'kanban_show'}, _kanban_worker_guidance='')
    monkeypatch.setenv('HERMES_KANBAN_TASK', 'later-task')
    assert _tool_guidance_block(agent) is None
