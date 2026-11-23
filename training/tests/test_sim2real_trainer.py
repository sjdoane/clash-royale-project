"""ppo_pool with the opt-in sim-to-real layer and the privileged critic: both are
off by default; the privileged critic never changes what the policy net sees and
its first update is the update without it; the layer reaches the workers and its
strength range can be widened while the run goes on.

    CR_ENGINE=build/gen2/engine .venv/bin/python -m pytest training/tests/test_sim2real_trainer.py -q

Same data needs as test_ppo_pool.py; skipped without them.
"""
import shutil

import numpy as np
import pytest
import torch

from training import pool_env as PE
from training import sim2real as S
from training.tests.test_ppo_pool import PP, _args, _cycle, _trainer, needs_data  # noqa: F401


def _env0(tr):
    return tr.envs.envs[0]


@needs_data
def test_everything_is_off_by_default(PP, tmp_path):
    a = _args(PP, tmp_path / "a")
    assert (a.sim2real, a.sim2real_scale, a.sim2real_only, a.sim2real_off) == ("", "", "", "")
    assert a.sim2real_widen == 0.0 and a.priv_critic is False
    tr = PP.PoolTrainer(a)
    assert tr._sim2real_spec() is None and tr.priv is None
    assert a.sim2real_anchor_clean is False and tr._clean is None
    for bad in (("--sim2real-widen", "5"), ("--sim2real", "v1", "--sim2real-widen", "5"),
                ("--sim2real-scale", "0:1"), ("--sim2real-anchor-clean",)):
        with pytest.raises(SystemExit):
            PP.PoolTrainer(_args(PP, tmp_path / "bad", *bad))._sim2real_spec()


@needs_data
def test_privileged_critic_leaves_the_policy_inputs_and_the_first_update_alone(PP, tmp_path):
    # One trainer at a time: each start reseeds the process-wide generators the
    # rollout and the update draw from.
    a = _trainer(PP, tmp_path / "a", "--critic-warmup", "0")
    b = c = d = None
    try:
        assert type(_env0(a)) is PE.PoolEnv and a.priv is None
        a.collect_rollout()
        ba = {k: v.clone() for k, v in a.buffer.stack().items()}
        obs_a = np.array(a._obs, copy=True)
        sa = a.run_update()

        b = _trainer(PP, tmp_path / "b", "--critic-warmup", "0", "--priv-critic")
        assert type(_env0(b)) is PE.PoolEnv and b.priv is not None
        assert not b.priv.g[-1].weight.any() and not b.priv.g[-1].bias.any()      # zero residual
        b.collect_rollout()
        bb = b.buffer.stack()
        # What the policy net was given and what it did: identical, entry for entry.
        assert set(ba) == set(bb)
        for k in ba:
            assert torch.equal(ba[k], bb[k]), k
        assert np.array_equal(obs_a, b._obs)
        T, n = ba["values"].shape
        assert len(b.priv.rows) == T and b.priv.rows[0].shape == (n, 18)
        # The stored privileged rows are the state each action was chosen in.
        assert np.array_equal(b.priv.cur, np.stack([e.privileged()[:18] for e in b.envs.envs]))
        assert np.array_equal(b.priv.rows[0][:, 12:18], np.ones((n, 6)))   # first row: untouched towers
        sb = b.run_update()
        assert sa.actor_loss == pytest.approx(sb.actor_loss, abs=1e-9)
        wa, wb = a.net.state_dict(), b.net.state_dict()
        assert list(wa) == list(wb)                                # no new parameter in the net
        for k in wa:
            assert torch.equal(wa[k], wb[k]), k                    # zero residual: the same update
        assert b.priv.g[-1].weight.abs().sum() > 0                 # ... and g has started to learn
        assert "ev_priv" in b._rec and "ev_priv" not in a._rec
        a.log_update(sa)
        b.log_update(sb)
        a.buffer.clear()
        b.buffer.clear()
        for tr in (a, b):
            tr._hx, tr._cx = tr._hx.detach(), tr._cx.detach()
            tr.periodic(None)
        _cycle(b)                                                  # an update with a live residual
        assert np.isfinite(b._rec["ev_priv"]) and np.isfinite(b._rec["priv_loss"])
        blob = torch.load(b.weight_path, map_location="cpu", weights_only=False)
        plain = torch.load(a.weight_path, map_location="cpu", weights_only=False)
        assert "priv_critic" in blob and "priv_critic" not in plain
        assert list(blob["model"]) == list(plain["model"])
        # resume with the flag: g comes back; a run started without it gets a zero g
        shutil.copytree(b.run_dir, str(tmp_path / "c"))
        c = _trainer(PP, tmp_path / "c", "--critic-warmup", "0", "--priv-critic")
        for k, v in b.priv.g.state_dict().items():
            assert torch.equal(v, c.priv.g.state_dict()[k]), k
        shutil.copytree(a.run_dir, str(tmp_path / "d"))
        d = _trainer(PP, tmp_path / "d", "--critic-warmup", "0", "--priv-critic")
        assert d.full_resume and not d.priv.g[-1].weight.any()
    finally:
        for tr in (a, b, c, d):
            if tr is not None:
                tr.close()


@needs_data
def test_the_layer_reaches_the_envs_and_widens(PP, tmp_path):
    tr = _trainer(PP, tmp_path / "a", "--sim2real", "v1", "--sim2real-scale", "0:1.5",
                  "--sim2real-widen", "4", "--priv-critic")
    try:
        env = _env0(tr)
        assert isinstance(env, S.pool_env_class()) and env.s2r is not None
        assert env.s2r.cfg.scale_range == (0.0, 0.0) and tr._s2r_now == [0.0, 0.0]
        assert env.s2r.cfg.enabled() == list(S.NAMES)
        _cycle(tr)
        tr.episodes_completed = max(tr.episodes_completed, 2)
        tr._widen_sim2real()
        assert tr._s2r_now == [0.0, pytest.approx(0.75)]
        for e in tr.envs.envs:
            assert e.s2r._pending.scale_range == (0.0, pytest.approx(0.75))
        tr.episodes_completed = 100
        tr._widen_sim2real()
        assert tr._s2r_now == [0.0, 1.5]
        _cycle(tr)                                                 # a rollout and an update at strength > 0
        assert tr._rec["sim2real_scale_range"] == [0.0, 1.5]
        assert np.isfinite(tr._rec["critic_loss"])
    finally:
        tr.close()


@needs_data
def test_the_reference_policy_reads_the_true_values_when_asked(PP, tmp_path):
    a, b = S.clean_span()
    # strength 0: the readings ARE the truth, so the recorded rows must be the
    # rollout's own span, row for row (alignment), and the update is the plain one
    tr = _trainer(PP, tmp_path / "z", "--critic-warmup", "0", "--sim2real", "v1", "--sim2real-scale", "0",
                  "--sim2real-anchor-clean")
    try:
        assert tr._clean is not None and _env0(tr).s2r.cfg.publish_clean
        tr.collect_rollout()
        obs = tr.buffer.stack()["obs"]
        rows = np.stack(tr._clean.rows)
        assert rows.shape == (obs.shape[0], obs.shape[1], b - a)
        assert np.array_equal(rows, obs[..., a:b].cpu().numpy())
        tr.run_update()
        assert tr._clean.rows == []
    finally:
        tr.close()
    # strength 1: the reference tables are the clone's on the clean rows, the
    # learner's rows keep the readings, and the two differ
    tr = _trainer(PP, tmp_path / "y", "--critic-warmup", "0", "--sim2real", "v1", "--sim2real-anchor-clean")
    seen = {}
    try:
        tr.collect_rollout()
        batch = {k: v.clone() for k, v in tr.buffer.stack().items()}
        rows = torch.as_tensor(np.stack(tr._clean.rows))
        assert not torch.equal(rows, batch["obs"][..., a:b].cpu())        # the layer did something
        inner = PP.policy_tables

        def tables(net, bt, *args, **kw):
            if net is tr.ref_net and "ref_obs" not in seen:
                seen["ref_obs"] = bt["obs"].detach().cpu().clone()
            return inner(net, bt, *args, **kw)
        PP.policy_tables = tables
        try:
            tr.run_update()
        finally:
            PP.policy_tables = inner
        ref_obs = seen["ref_obs"]
        assert torch.equal(ref_obs[..., a:b], rows)                       # the clone saw the truth
        assert torch.equal(ref_obs[..., :a], batch["obs"][..., :a].cpu())  # and the same board
        assert torch.equal(ref_obs[..., b:], batch["obs"][..., b:].cpu())
    finally:
        tr.close()


@needs_data
def test_estimator_share_and_full_health_board_reach_the_envs(PP, tmp_path):
    plain = _trainer(PP, tmp_path / "p", "--sim2real", "v1")
    try:
        assert _env0(plain)._noise.cfg.hp_as_live is False
        assert _env0(plain).s2r.cfg.elixir_est is False and _env0(plain).s2r.cfg.elixir_est_share == 1.0
    finally:
        plain.close()
    tr = _trainer(PP, tmp_path / "a", "--sim2real", "v1", "--sim2real-scale", "0:1.5", "--priv-critic",
                  "--sim2real-elixir-est", "--sim2real-elixir-est-share", "0.8", "--board-hp-live")
    try:
        for e in tr.envs.envs:
            assert e.s2r.cfg.elixir_est is True and e.s2r.cfg.elixir_est_share == 0.8
            assert e._noise.cfg.hp_as_live is True
        _cycle(tr)
        assert np.isfinite(tr._rec["critic_loss"])
    finally:
        tr.close()
    with pytest.raises(SystemExit):
        _trainer(PP, tmp_path / "bad", "--board-hp-live", "--no-board-noise")
