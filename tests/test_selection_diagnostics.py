import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

from dms.policy import IndependentAgent

spec = importlib.util.spec_from_file_location("selection_probe", Path(__file__).resolve().parents[1] / "scripts/diagnose_intentions.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def linear_agent():
    agent = IndependentAgent(1, 1, 3, latent_dim=1, coordination="causal", observer_id=0)
    agent.actor = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        agent.actor.weight.copy_(torch.tensor([[0., 99., 1., 2.]]))
        agent.log_std.zero_()
    return agent


def test_selection_probe_matches_known_gaussian_kl_without_mutating_inputs():
    agent = linear_agent()
    obs = torch.zeros(2, 1)
    context = torch.tensor([[[9.], [1.], [2.]], [[99.], [1.], [2.]]])
    visible = torch.ones(2, 3, dtype=torch.bool)
    before_context, before_visible = context.clone(), visible.clone()
    rows = probe.selection_diagnostics(agent, obs, context, visible, None, .6)
    assert [row['peer'] for row in rows] == [1, 2]
    assert [row['visible_policy_count'] for row in rows] == [2, 2]
    assert rows[0]['mean_kl'] == pytest.approx(.5)
    assert rows[1]['mean_kl'] == pytest.approx(8.)
    assert rows[0]['fraction_above_threshold']['0.6'] == 0
    assert rows[1]['fraction_above_threshold']['0.6'] == 1
    torch.testing.assert_close(context, before_context)
    torch.testing.assert_close(visible, before_visible)
    assert all(parameter.grad is None for parameter in agent.parameters())


def test_selection_probe_excludes_unobserved_peers_and_respects_strict_threshold():
    agent = linear_agent()
    obs = torch.zeros(2, 1)
    context = torch.tensor([[[999.], [1.], [9999.]], [[999.], [3.], [-9999.]]])
    visible = torch.tensor([[True, True, False], [True, False, False]])
    rows = probe.selection_diagnostics(agent, obs, context, visible, None, .5)
    assert rows[0]['visible_policy_count'] == 1
    assert rows[0]['mean_kl'] == pytest.approx(.5)
    assert rows[0]['fraction_above_threshold']['0.5'] == 0
    assert rows[1]['visible_policy_count'] == 0
    assert rows[1]['mean_kl'] is rows[1]['quantiles'] is None
