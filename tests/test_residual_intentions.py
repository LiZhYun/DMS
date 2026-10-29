"""Residual intentions preserve the local-policy starting point and boundaries."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dms.policy import IndependentAgent
from dms.train import Config, Learner, collect, create_env, empty_history, seed_everything, update


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def environment(discrete):
    return SimpleNamespace(n_agents=3, action_dim=3, action_dims=[3]*3 if discrete else [1,2,3],
        obs_dim=4, discrete=discrete, action_visibility=np.ones((3,3,3), bool))


def observations(length, batch=5):
    if not length:
        return torch.randn(batch,4)
    windows = torch.randn(batch,length,5)
    windows[:,:,-1] = 1
    windows[0,:2,-1] = 0
    return windows


def assert_same(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_same(x,y)
    else:assert a==b


@pytest.mark.parametrize('coordination',['soft','all','causal'])
@pytest.mark.parametrize('history',[0,4])
@pytest.mark.parametrize('discrete',[False,True])
def test_initial_local_parameters_rng_actions_and_values_match_ippo(coordination,history,discrete):
    common=dict(hidden_dim=8,latent_dim=2,coordination=coordination,policy_history_length=history)
    torch.manual_seed(31)
    baseline=Learner(Config(algorithm='ippo',**common),environment(discrete))
    baseline_rng=torch.get_rng_state().clone()
    torch.manual_seed(31)
    residual=Learner(Config(intention_fusion='residual',**common),environment(discrete))
    assert_same(baseline_rng,torch.get_rng_state())
    for left,right in zip(baseline.agents,residual.agents):
        for key,value in left.state_dict().items():assert_same(value,right.state_dict()[key])
        obs=observations(history);context=torch.randn(5,3,2);gates=torch.ones(5,3,dtype=torch.bool)
        mask=torch.tensor([[True,False,True]]).expand(5,-1) if discrete else None
        torch.manual_seed(55);expected=left.act(obs,context[:,:,:0],gates,mask)
        torch.manual_seed(55);actual=right.act(obs,context,gates,mask)
        assert_same(expected,actual)


@pytest.mark.parametrize('coordination',['soft','all','causal'])
def test_learned_projection_uses_visible_peers_and_zero_input_is_local_only(coordination):
    agent=IndependentAgent(4,3,3,2,hidden_dim=8,discrete=True,
                           coordination=coordination,intention_fusion='residual')
    with torch.no_grad():
        agent.actor_intention_projection.weight.normal_(std=.3)
        agent.critic_intention_projection.weight.normal_(std=.3)
    obs=torch.randn(5,4);context=torch.randn(5,3,2)
    gates=torch.tensor([[False,True,False]]).expand(5,-1)
    before=agent.distribution(obs,context,gates).probs
    changed=context.clone();changed[:,2]=float('nan')
    assert_same(before,agent.distribution(obs,changed,gates).probs)
    assert_same(agent.value(obs,context,gates),agent.value(obs,changed,gates))
    changed=context.clone();changed[:,1]+=5
    assert not torch.allclose(before,agent.distribution(obs,changed,gates).probs)
    absent=torch.zeros_like(gates)
    expected_logits=agent.actor(obs)
    expected=expected_logits-expected_logits.logsumexp(-1,keepdim=True)
    assert_same(expected,agent.distribution(obs,context,absent).logits)
    assert_same(agent.critic(obs).squeeze(-1),agent.normalized_value(obs,context,absent))


def test_critic_projection_has_critic_lr_and_actor_projection_has_actor_lr():
    learner=Learner(Config(intention_fusion='residual',hidden_dim=8,latent_dim=2),environment(True))
    for agent,optimizer in zip(learner.agents,learner.optimizers):
        actor_ids={id(p) for p in optimizer.param_groups[0]['params']}
        critic_ids={id(p) for p in optimizer.param_groups[1]['params']}
        assert id(agent.actor_intention_projection.weight) in actor_ids
        assert id(agent.critic_intention_projection.weight) in critic_ids
        assert not actor_ids & critic_ids
        obs=torch.randn(5,4);context=torch.randn(5,3,2);gates=torch.ones(5,3,dtype=torch.bool)
        agent.value(obs,context,gates).square().mean().backward()
        assert agent.actor_intention_projection.weight.grad is None
        assert agent.critic_intention_projection.weight.grad.abs().sum()>0
        agent.zero_grad(set_to_none=True)
        agent.distribution(obs,context,gates).log_prob(torch.zeros(5,dtype=torch.long)).mean().backward()
        assert agent.critic_intention_projection.weight.grad is None
        assert agent.actor_intention_projection.weight.grad.abs().sum()>0


def test_popart_preserves_value_after_residual_projection_has_learned():
    agent=IndependentAgent(4,2,3,2,hidden_dim=8,intention_fusion='residual')
    with torch.no_grad():agent.critic_intention_projection.weight.normal_()
    obs=torch.randn(9,4);context=torch.randn(9,3,2);gates=torch.ones(9,3,dtype=torch.bool)
    before=agent.value(obs,context,gates).detach()
    agent.update_value_scale(torch.linspace(100,300,100))
    torch.testing.assert_close(before,agent.value(obs,context,gates),atol=4e-5,rtol=4e-5)


@pytest.mark.parametrize('mode',['independent','sequential'])
def test_joint_update_learns_projection_and_keeps_future_targets_out_of_policy(mode):
    from test_intention_rl import joint_batch, joint_model
    a=joint_model(mode=mode,intention_fusion='residual')
    data=joint_batch(a);b=deepcopy(a);changed=deepcopy(data)
    changed['history'][~changed['visible']]=float('nan')
    changed['future'][:]=float('nan')
    for learner,batch in [(a,data),(b,changed)]:
        torch.manual_seed(61);update(learner,batch)
    assert_same(a.agents.state_dict(),b.agents.state_dict())
    assert_same(a.intentions.state_dict(),b.intentions.state_dict())
    for agent in a.agents:
        assert agent.actor_intention_projection.weight.abs().sum()>0
        assert agent.critic_intention_projection.weight.abs().sum()>0


def test_first_active_rollout_matches_ippo_before_any_learning():
    recorded=[]
    for algorithm in ['ippo','dms']:
        c=Config(algorithm=algorithm,intention_fusion='residual',num_envs=1,
                 hidden_dim=8,latent_dim=2,history_length=4,prediction_horizon=2,
                 rollout_steps=40,total_steps=40,warmup_episodes=0,policy_history_length=4)
        seed_everything(71);env=create_env(c)
        try:
            learner=Learner(c,env);history,visible=empty_history(learner,1)
            data,*_=collect(learner,[env],[env.reset(seed=71)],history,visible,0,np.zeros(1))
            recorded.append(data)
        finally:env.close()
    assert recorded[1]['gates'].any()
    for key in ['obs','raw_obs','actions','log_prob','values','rewards','next_values','terminated','truncated']:
        np.testing.assert_array_equal(recorded[0][key],recorded[1][key])


def test_residual_checkpoint_roundtrip_preserves_outputs(tmp_path):
    c=Config(intention_fusion='residual',hidden_dim=8,latent_dim=2)
    learner=Learner(c,environment(True))
    for agent in learner.agents:
        with torch.no_grad():
            agent.actor_intention_projection.weight.normal_()
            agent.critic_intention_projection.weight.normal_()
    path=tmp_path/'residual.pt';learner.save(path,128,4)
    checkpoint=torch.load(path,weights_only=False);restored=Learner(Config(**checkpoint['config']),environment(True))
    restored.agents.load_state_dict(checkpoint['agents']);restored.intentions.load_state_dict(checkpoint['intentions'])
    obs=torch.randn(5,3,4);context=torch.randn(5,3,3,2);gates=torch.ones(5,3,3,dtype=torch.bool)
    assert_same(learner.action(obs,context,gates,None,True),restored.action(obs,context,gates,None,True))


@pytest.mark.parametrize('fusion',[None,True,'add',''])
def test_invalid_fusion_is_rejected(fusion):
    with pytest.raises(ValueError,match='intention_fusion'):Config(intention_fusion=fusion).validate()
    with pytest.raises(ValueError,match='intention_fusion'):IndependentAgent(4,2,3,2,intention_fusion=fusion)
