"""The gen-2 PPO trainer (training/ppo_pool.py): the per-decision constants keep
their meaning in game seconds, the KL-to-clone term is zero at the start and
positive once the policy moves, the factorised log-probability the update
recomputes is the one the rollout stored, snapshots reach the workers' pool,
and a resumed run makes the same next update.

    CR_ENGINE=build/gen2/engine .venv/bin/python -m pytest training/tests/test_ppo_pool.py -q

The trainer tests need the ghost cache (data/ghost/v1), the stand-in clone
(runs/bc3/p_L0_H10/last.pth), the pilot dataset (data/bc2/pilot) and the gen-1
league checkpoint the default pool names; they are skipped without them.
"""
import copy
import json
import math
import os
import shutil

import numpy as np
import pytest
import torch

from training import common as C  # noqa: F401  (sets CLASH_DECK first)
from training import ghost as G

INIT = os.path.join(C.REPO, "runs", "bc3", "p_L0_H10", "last.pth")
PILOT = os.path.join(C.REPO, "data", "bc2", "pilot")
SELF = os.path.join(C.REPO, "remote_results", "ppo_league26_ep138708.pth")
HAVE = (os.path.exists(os.path.join(G.STORE, "meta.json")) and os.path.exists(INIT)
        and os.path.exists(os.path.join(PILOT, "dataset.json")) and os.path.exists(SELF))
needs_data = pytest.mark.skipif(not HAVE, reason="needs the ghost cache, the stand-in clone, "
                                                 "the pilot dataset and remote_results")


@pytest.fixture(scope="module")
def PP():
    """The module, imported late (it sets the engine's environment switches),
    with the engine trainer's reward hooks put back afterwards."""
    from training import ppo_pool
    bt = ppo_pool.bt
    names = ("extract_engine_stats", "compute_shaping", "flawless_defense_bonus", "DRAW_PENALTY")
    keep = {n: getattr(bt, n) for n in names}
    yield ppo_pool
    for n, v in keep.items():
        setattr(bt, n, v)


def _args(PP, out, *extra):
    return PP._parse(["--run", "t", "--out", str(out), "--init", INIT, "--decision-ticks", "5",
                      "--num-envs", "2", "--vec", "sync", "--rollout-steps", "48", "--chunk-steps", "12",
                      "--minibatches", "2", "--epochs", "2", "--critic-warmup", "1", "--bc-batch", "64",
                      "--bc-data", PILOT, "--seed", "5", "--torch-threads", "0", *extra])


def _trainer(PP, out, *extra):
    tr = PP.PoolTrainer(_args(PP, out, *extra))
    tr.setup()
    return tr


def _cycle(tr):
    """One pass of BaseTrainer.run's loop body."""
    tr.collect_rollout()
    stats = tr.run_update()
    tr.log_update(stats)
    tr.buffer.clear()
    tr._hx, tr._cx = tr._hx.detach(), tr._cx.detach()
    tr.periodic(stats)
    return stats


# ------------------------------------------------------------ constants ---
def test_constants_keep_their_meaning_in_seconds(PP):
    ten = PP.convert_constants(10)
    assert ten["gamma"] == pytest.approx(0.999) and ten["gae_lambda"] == pytest.approx(0.9)
    assert (ten["update_timestep"], ten["bptt_chunk"]) == (500, 50)        # gen 1's own values
    five = PP.convert_constants(5)
    assert (five["update_timestep"], five["bptt_chunk"]) == (1000, 100)
    assert five["rollout_seconds"] == ten["rollout_seconds"] == 500.0
    assert five["chunk_seconds"] == ten["chunk_seconds"] == 50.0
    for dt in (2, 5, 10, 20):
        c = PP.convert_constants(dt, rollout_seconds=400, chunk_seconds=40)
        per_s = 10.0 / dt                                                   # decisions per game second
        assert c["gamma"] ** per_s == pytest.approx(0.999, abs=1e-12)
        assert c["gae_lambda"] ** per_s == pytest.approx(0.9, abs=1e-12)
        # the weight GAE gives a TD error one game second later
        assert (c["gamma"] * c["gae_lambda"]) ** per_s == pytest.approx(0.999 * 0.9, abs=1e-12)
        assert c["update_timestep"] * dt == 4000 and c["bptt_chunk"] * dt == 400
    # a reward 30 s away is discounted alike at either interval
    assert five["gamma"] ** 60 == pytest.approx(ten["gamma"] ** 30, abs=1e-12)
    assert five["gae_horizon_seconds"] == pytest.approx(ten["gae_horizon_seconds"], rel=0.05)
    assert PP.convert_constants(5, rollout_steps=16, chunk_steps=8)["update_timestep"] == 16
    with pytest.raises(ValueError):
        PP.convert_constants(5, rollout_seconds=500, chunk_seconds=30)
    # 12 plays in 600 decisions of 5 ticks = 5 game minutes
    assert PP.per_minute(12, 600, 5) == pytest.approx(2.4)
    assert PP.per_minute(12, 600, 10) == pytest.approx(1.2)


# ----------------------------------------------------- factorised KL ---
def _composed(PP, n=64, hand=4, seed=0, shift=0.0):
    g = torch.Generator().manual_seed(seed)
    when = torch.randn(n, generator=g) - 1.0 + shift
    what = torch.randn(n, hand, generator=g) * (1.0 + shift)
    g = torch.Generator().manual_seed(99)                 # the legality mask is the state's
    legal = torch.rand(n, hand, generator=g) > 0.35
    legal[:4] = False                                     # some rows can only wait
    what = what.masked_fill(~legal, float("-inf"))
    return PP.N3.NetV3.compose_card_logits(when, what, legal.any(-1)), legal


def test_kl_heads_zero_then_positive_and_adds_up(PP):
    lp, legal = _composed(PP)
    lq, _ = _composed(PP, seed=1, shift=0.3)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(lp.shape[0]), atol=1e-5)
    kw, kh, pp = PP.kl_heads(lp, lp)
    assert float(kw.abs().max()) < 1e-6 and float(kh.abs().max()) < 1e-6
    kw, kh, pp = PP.kl_heads(lp, lq)
    has = legal.any(-1)
    assert float(kw[~has].abs().max()) == 0.0 and float(kh[~has].abs().max()) == 0.0
    assert float(kw[has].min()) > 0.0
    multi = legal.sum(-1) > 1
    assert float(kh[multi].min()) > 0.0
    assert float(kh[has & ~multi].abs().max()) < 1e-6      # one legal card: nothing to choose
    # KL of the composed (slots, wait) distributions = WHEN + p_play * WHAT
    ok = torch.isfinite(lp)
    direct = (lp.exp() * torch.where(ok, lp - lq, torch.zeros_like(lp))).sum(-1)
    assert torch.allclose(kw + pp * kh, direct, atol=1e-5)
    hw, hh, pp2 = PP.head_entropies(lp)
    comp = -(lp.exp() * torch.where(ok, lp, torch.zeros_like(lp))).sum(-1)
    assert torch.allclose(hw + pp2 * hh, comp, atol=1e-5)
    assert torch.allclose(pp, pp2)


# ------------------------------------------------------- pool plumbing ---
def test_self_entries_weights(PP):
    assert PP.self_entries("/r/latest.pth") == [{"path": "/r/latest.pth", "weight": 1.0, "name": "latest"}]
    older = [{"path": f"/r/snapshots/ep{i:07d}.pth", "name": f"snap_ep{i}"} for i in range(1, 21)]
    seeds = [{"path": "/x/gen1.pth", "name": "gen1_league", "weight": 9.0}]
    e = PP.self_entries("/r/latest.pth", older, latest_share=0.5, max_older=8, seeds=seeds)
    assert e[0] == {"path": "/r/latest.pth", "weight": 0.5, "name": "latest"}
    assert sum(x["weight"] for x in e) == pytest.approx(1.0)
    assert len(e) == 1 + 1 + 8 and e[1]["name"] == "gen1_league"
    assert len({round(x["weight"], 9) for x in e[1:]}) == 1          # uniform over the older ones
    names = [x["name"] for x in e[2:]]
    assert names[0] == "snap_ep1" and names[-1] == "snap_ep20" and len(set(names)) == 8
    assert PP.thin(range(5), 8) == [0, 1, 2, 3, 4] and PP.thin(range(10), 1) == [9]
    assert PP.ckpt_name("runs/bc3/g2_big_L0_H10/last.pth") == "g2_big_L0_H10"
    assert PP.ckpt_name("remote_results/bc_hog26.pth") == "bc_hog26"


def test_build_pool_clone_entry_and_fallback(PP):
    base = os.path.join("training", "pool_gen2_default.json")
    pool, note = PP.build_pool(base)
    w = pool["weights"]
    assert w["clone"] == 0.0 and w["ghost"] == pytest.approx(0.70) and pool["clone"]["ckpts"] == []
    assert "clone weight 0.35" in note and "ghost" in note
    assert sum(w.values()) == pytest.approx(1.0)
    pool, note = PP.build_pool(base, clone_fallback="self")
    assert pool["weights"]["self"] == pytest.approx(0.55) and "self" in note
    pool, note = PP.build_pool(base, clone_ckpts=["runs/bc3/g2_big_L0_H10/last.pth", "runs/bc3/x/last.pth"])
    assert note == "" and pool["weights"]["clone"] == pytest.approx(0.35)
    assert [c["name"] for c in pool["clone"]["ckpts"]] == ["g2_big_L0_H10", "x"]
    assert all(os.path.isabs(c["path"]) for c in pool["clone"]["ckpts"])
    pool, note = PP.build_pool(base, {"weights": {"teacher": 0.2}, "p_midmatch": 0.1})
    assert pool["weights"]["teacher"] == 0.2 and pool["p_midmatch"] == 0.1
    with pytest.raises(ValueError):                         # the validation split is the evaluation set
        PP.build_pool(base, {"ghost": {"split": "val"}})


def test_guard_reasons(PP):
    ok = {"Hog_Rider": 0.9, "Fireball": 1.1}
    assert PP.guard_reasons(1.0, ok, 1.0) == []
    assert PP.guard_reasons(1.0, ok, None) == []
    assert len(PP.guard_reasons(0.5, ok, 1.0)) == 1
    assert len(PP.guard_reasons(1.7, ok, 1.0)) == 1
    r = PP.guard_reasons(1.0, {"Hog_Rider": 0.9, "Fireball": 0.3, "The_Log": 0.4}, 1.0)
    assert len(r) == 2 and r[0].startswith("Fireball")
    assert len(PP.guard_reasons(1.0, ok, 2.5)) == 1 and len(PP.guard_reasons(1.0, ok, 0.4)) == 1
    assert len(PP.guard_reasons(0.5, {"Fireball": 0.1}, 0.2)) == 3


# -------------------------------------------------------------- trainer ---
@needs_data
def test_rollout_logprob_kl_and_warmup(PP, tmp_path):
    tr = _trainer(PP, tmp_path / "a")
    try:
        net = tr.net
        ck = torch.load(INIT, map_location="cpu", weights_only=False)
        assert net.cfg == ck["cfg"] and tr.dt == 5                 # net config from the checkpoint
        assert tr.cfg.update_timestep == 48 and tr.cfg.gamma == pytest.approx(0.999 ** 0.5)
        assert tr.pool["weights"]["clone"] == 0.0 and "moved to ghost" in tr.pool_note

        # -- the stored log-probability is the factorised one -------------------
        tr.collect_rollout()
        batch = tr.buffer.stack()
        T, N = batch["card_actions"].shape
        tab = PP.policy_tables(net, batch)
        lplay, lwait, lwhat, legal, has = PP.card_factors(tab["card"])
        cards, cells = batch["card_actions"], batch["placement_actions"]
        idle = batch["ability_ready"].reshape(T, N, -1).sum(-1) == 0   # no ability term in the row
        dec = batch["decision"] > 0
        play = dec & idle & (cards != net.hand_size)
        wait = dec & idle & (cards == net.hand_size)
        assert int(play.sum()) >= 3 and int(wait.sum()) >= 10
        slot = cards.clamp(max=net.hand_size - 1).unsqueeze(-1)
        expect_play = (lplay + lwhat.gather(-1, slot).squeeze(-1)
                       + tab["place"].gather(-1, cells.unsqueeze(-1)).squeeze(-1))
        assert torch.allclose(batch["logprobs"][play], expect_play[play], atol=2e-4)
        # a wait carries a constant placement term: uniform over the no-op's legal cells
        flat = batch["obs"].reshape(T * N, -1)
        with torch.no_grad():
            f, emb, sp, hi = net.extract_features_hires(flat)
            o = net.heads(f, net.affordability_mask(flat))
            raw = PP.N3.NetV3._placement_raw(net, o["hx"], emb, torch.full((T * N,), net.hand_size),
                                             flat, sp, hires_map=hi, cycle_feat=net._split_cycle(f))
        n_cells = torch.isfinite(raw).sum(-1).view(T, N).float()
        expect_wait = lwait - torch.log(n_cells)
        assert torch.allclose(batch["logprobs"][wait], expect_wait[wait], atol=2e-4)

        # -- KL to the clone: zero at the start ---------------------------------
        diag, ratios = tr.diagnose(batch)
        assert all(abs(v) < 1e-6 for v in diag["kl_pre"].values()), diag["kl_pre"]
        assert diag["play_ratio"] == pytest.approx(1.0, abs=1e-5)
        assert all(abs(r - 1.0) < 1e-4 for r in ratios.values())
        assert sum(c["plays"] for c in diag["cards"].values()) == diag["sampled_plays"]

        # -- critic warm-up: the engine's recomputation agrees, the policy does not move
        before = copy.deepcopy(net.state_dict())
        stats = tr.run_update()
        rec = tr._rec
        assert rec["warmup"] and rec["update"] == 1
        assert rec["ratio_dev_first"] < 1e-3                       # stored == recomputed, every row
        assert abs(rec["kl_first"]) < 1e-6 and rec["bc_loss"] is None
        assert math.isfinite(stats.critic_loss) and stats.nonfinite_skips == 0
        moved = {k for k, v in net.state_dict().items() if not torch.equal(v, before[k])}
        assert moved and all(k.startswith("value_head") for k in moved), sorted(moved)[:5]
        assert all(p.requires_grad for p in net.parameters())
        tr.log_update(stats)
        tr.buffer.clear()
        tr.periodic(stats)

        # -- first policy update: starts at KL 0, ends away from the clone -------
        stats = _cycle(tr)
        rec = tr._rec
        assert not rec["warmup"] and abs(rec["kl_first"]) < 1e-6
        assert rec["ratio_dev_first"] < 1e-3
        assert rec["kl_upd"]["kl"] > 0 and rec["bc_loss"] is not None and math.isfinite(rec["bc_loss"])
        assert rec["bc_coef"] == pytest.approx(0.5, rel=1e-2) and rec["kl_coef"] == pytest.approx(1.0, rel=1e-2)
        assert all(math.isfinite(rec[k]) for k in ("actor_loss", "critic_loss", "plays_per_min"))
        assert rec["guard"]["ppm_ratio"] is not None and rec["plays_per_min_clone_baseline"] > 0

        # -- positive after a perturbation, in every head, in the update's own term
        with torch.no_grad():
            for name, p in net.named_parameters():
                if not name.startswith("value_head"):
                    p.add_(0.02 * torch.randn_like(p))
        tr.collect_rollout()
        diag, _ = tr.diagnose(tr.buffer.stack())
        for head in ("when_r", "what_r", "where_r", "when_f", "what_f", "where_f"):
            assert diag["kl_pre"][head] > 1e-6, (head, diag["kl_pre"])
        stats = tr.run_update()
        assert tr._rec["kl_first"] > 1e-6 and tr._rec["kl_upd"]["when_r"] > 0
        tr.log_update(stats)
        tr.buffer.clear()

        lines = [json.loads(x) for x in open(tr.progress_path)]
        assert [x["update"] for x in lines] == [1, 2, 3]
        for key in ("actor_loss", "critic_loss", "kl_pre", "kl_upd", "bc_loss", "entropy", "plays_per_min",
                    "cards", "elixir_leaked_per_min", "by_kind", "mix", "episodes_per_s", "guard", "pool_note"):
            assert key in lines[-1], key
    finally:
        tr.close()


@needs_data
def test_snapshot_reaches_the_pool_and_checkpoints_load(PP, tmp_path):
    from training import obs_compat
    tr = _trainer(PP, tmp_path / "s", "--clone-fallback", "self", "--pool-override",
                  json.dumps({"weights": {"ghost": 0, "teacher": 0, "self": 1.0}}))
    try:
        first = tr.envs.call("get_pool")[0]["self"]["ckpts"]
        assert [e["name"] for e in first] == ["latest", "gen1_league"]
        assert [e["weight"] for e in first] == [0.5, 0.5]
        tr.episodes_completed = 7
        path = tr.take_snapshot()
        assert os.path.basename(path) == "ep0000007.pth" and os.path.exists(path)
        tr.save_checkpoint()
        for pool in tr.envs.call("get_pool"):                       # every worker has the new entry
            entries = {e["name"]: e for e in pool["self"]["ckpts"]}
            assert set(entries) == {"latest", "gen1_league", "snap_ep7"}
            assert entries["latest"]["weight"] == 0.5 and entries["latest"]["path"] == tr.latest_path
            assert entries["snap_ep7"]["weight"] == 0.25 and entries["snap_ep7"]["path"] == path
        # ... and plays it: the pool takes effect at the next reset
        seen = set()
        for s in range(12):
            _, infos = tr.envs.reset(seed=[100 + 2 * s, 101 + 2 * s])
            seen |= {str(n) for n in infos["opp_ckpt"]}
            assert all(int(k) == PP.PE.SELF for k in infos["opp_kind"])
        assert "snap_ep7" in seen and "latest" in seen, seen
        # both files are policy checkpoints the evaluation and the live driver read
        for p in (path, tr.latest_path):
            pol, info = obs_compat.load_policy(p)
            assert info["decision_ticks"] == 5 and info["names"] == tr.init_blob["names"]
            assert info["lead"] == tr.init_blob["lead"] and info["cfg"] == tr.net_cfg
            net, meta = PP.N3.load_v3(p)
            assert meta["ppo"]["episodes"] == 7 and type(net) is PP.N3.NetV3
            for k, v in tr.net.state_dict().items():
                assert torch.equal(v, net.state_dict()[k]), k
    finally:
        tr.close()


@needs_data
def test_resume_makes_the_same_next_update(PP, tmp_path):
    a = _trainer(PP, tmp_path / "a")
    b = None
    try:
        _cycle(a)                                   # update 1: critic warm-up
        _cycle(a)                                   # update 2: policy, anchors, cloning rows
        a.collect_rollout()
        batch = {k: v.clone() for k, v in a.buffer.stack().items()}
        obs = np.array(a._obs, copy=True)
        a.save_checkpoint()                         # what a stopped run leaves behind
        shutil.copytree(a.run_dir, str(tmp_path / "b"))
        sa = a.run_update()
        ra = a._rec

        b = _trainer(PP, tmp_path / "b")            # same command, new directory: resumes
        assert b.full_resume and b.n_updates == 2 and b.episodes_completed == a.episodes_completed
        assert b.cfg.seed != a.cfg.seed             # new episodes, not a replay of the first start's
        assert b.baseline == a.baseline and b.guard["streak"] == 0
        b.buffer.drain = lambda: batch
        b._obs = obs
        b._rollout_s = 1.0
        sb = b.run_update()
        rb = b._rec
        assert rb["update"] == ra["update"] == 3
        for key in ("actor_loss", "critic_loss", "kl_first", "bc_loss", "kl_coef", "bc_coef"):
            assert rb[key] == pytest.approx(ra[key], rel=1e-6, abs=1e-9), key
        assert sb.clip_frac == pytest.approx(sa.clip_frac)
        wa, wb = a.net.state_dict(), b.net.state_dict()
        for k in wa:
            assert torch.allclose(wa[k], wb[k], rtol=0, atol=1e-7), k
        # the resumed log continues the same file
        b.log_update(sb)
        assert [json.loads(x)["update"] for x in open(b.progress_path)] == [1, 2, 3]
    finally:
        a.close()
        if b is not None:
            b.close()


def test_kl_heads_ignores_rows_where_only_one_side_can_play(PP):
    """With --sim2real-anchor-clean the clone's legality comes from the true
    values and the learner's from its readings: a row where one of them is a
    forced wait must add nothing to WHEN or WHAT, in either direction, and must
    not send a gradient (2026-11-22: such rows gave a WHEN KL of 1e24)."""
    lp, legal = _composed(PP)
    lq, _ = _composed(PP, seed=1, shift=0.3)
    forced = lq.clone()
    forced[10:20, :-1] = float("-inf")                    # the reference can only wait here
    forced[10:20, -1] = 0.0
    lp = lp.clone().requires_grad_(True)
    for a, b in ((lp, forced), (forced, lp)):
        kw, kh, pp = PP.kl_heads(a, b)
        assert torch.isfinite(kw).all() and torch.isfinite(kh).all()
        assert float(kw[10:20].abs().max()) == 0.0 and float(kh[10:20].abs().max()) == 0.0
        assert float(kw.abs().max()) < 50.0
        g, = torch.autograd.grad((kw + pp.detach() * kh).sum(), lp, allow_unused=True)
        assert g is None or (torch.isfinite(g).all() and float(g[10:20].abs().max()) == 0.0)


def test_when_ref_exempts_decisions_at_a_full_bar(PP):
    """--anchor-free-elixir: a decision read at E elixir or more has no WHEN reference."""
    import types
    from training import common as C2
    obs = torch.zeros(4, 2, C2.SPATIAL_SIZE + 4)
    for i, e in enumerate((1.0, 0.95, 0.9, 0.3)):
        obs[i, :, C2.SPATIAL_SIZE] = e
    off = types.SimpleNamespace(args=types.SimpleNamespace(anchor_free_elixir=None))
    assert PP.PoolTrainer.when_ref(off, obs) is None
    on = types.SimpleNamespace(args=types.SimpleNamespace(anchor_free_elixir=9.5))
    w = PP.PoolTrainer.when_ref(on, obs)
    assert w.shape == (4, 2)
    assert w[:, 0].tolist() == [0.0, 0.0, 1.0, 1.0]
