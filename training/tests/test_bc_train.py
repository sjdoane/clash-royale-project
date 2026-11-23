"""The behaviour-cloning trainer for NetV3 (training/bc_train.py): targets, the
noise cache, the losses, label metrics, checkpoint metadata.

    CR_ENGINE=build/gen2/engine .venv/bin/python -m pytest training/tests/test_bc_train.py -q -p no:cacheprovider
"""
import argparse
import os
from collections import Counter

import numpy as np
import pytest
import torch

from training import common as C  # noqa: F401  (sets CLASH_DECK first)
from training import obs_compat as OC

pytestmark = pytest.mark.skipif(not OC.IS_GEN2, reason="NetV3 reads the gen-2 observation")
if OC.IS_GEN2:
    from training import bc_train as T
    from training import bcdata2 as D
    from training import net_v3 as N3
    from training import obs_v2 as OV

PILOT = os.path.join(C.REPO, "data", "bc2", "pilot")
COST = np.full(200, 99.0)
COST[[7, 8, 9, 10]] = [4.0, 2.0, 1.0, 3.0]
HAND = (7, 8, 9, 10)


def _ep(spec, hand=HAND, elixir=10.0):
    """One episode from (tick, kind, card, tile[, extra dict]) rows in time order."""
    rows = np.zeros(len(spec), D.ROW)
    rows["slot"], rows["tile"], rows["card"] = D.WAIT, -1, -1
    rows["hand"], rows["elixir"] = hand, elixir
    for i, s in enumerate(spec):
        rows["tick"][i], rows["kind"][i] = s[0], s[1]
        if s[1] & D.K_PLAY:
            rows["card"][i], rows["tile"][i] = s[2], s[3]
            rows["slot"][i] = list(hand).index(s[2]) if s[2] in hand else 0
        for k, v in (s[4] if len(s) > 4 else {}).items():
            rows[k][i] = v
    eps = np.zeros(1, D.EP)
    eps["row0"], eps["nrows"] = 0, len(spec)
    return rows, eps


def _targets(spec, lead, interval=10, **kw):
    rows, eps = _ep(spec, **kw)
    st = Counter()
    return T.build_targets(rows, eps, lead, interval, COST, st), st, rows


G, P = 4, 1          # D.K_GRID, D.K_PLAY


def test_constants_match_bcdata2():
    assert (D.K_GRID, D.K_PLAY) == (G, P)


def test_lead_zero_assigns_the_play_to_the_last_grid_row_before_it():
    t, st, _ = _targets([(0, G), (10, G), (13, P, 9, 77), (20, G), (30, G)], lead=0)
    assert t["when"].tolist() == [0, 1, -1, 0, 0]          # the play row itself is not a WHEN row
    assert (t["w_slot"][1], t["w_tile"][1], t["w_dt"][1]) == (2, 77, 3)
    assert st["play_assigned"] == 1 and st["plays"] == 1


def test_a_play_on_a_grid_tick_is_that_row():
    t, _, _ = _targets([(0, G), (10, G), (20, G | P, 7, 5), (30, G)], lead=0)
    assert t["when"].tolist() == [0, 0, 1, 0] and t["w_dt"][2] == 0 and t["w_slot"][2] == 0


def test_lead_moves_the_decision_earlier():
    spec = [(0, G), (10, G), (13, P, 9, 77), (20, G), (30, G)]
    t, _, _ = _targets(spec, lead=5)                        # 13 - 5 = 8: the row at tick 0
    assert t["when"].tolist() == [1, 0, -1, 0, 0] and t["w_dt"][0] == 13
    t, _, _ = _targets(spec, lead=3)                        # 13 - 3 = 10: the row at tick 10
    assert t["when"].tolist() == [0, 1, -1, 0, 0] and t["w_dt"][1] == 3


def test_a_play_not_playable_at_its_lead_row_moves_to_the_next_row():
    # at tick 0 the card is not affordable; at tick 10 it is
    spec = [(0, G, 0, 0, {"elixir": 0.5}), (10, G), (13, P, 9, 77), (20, G)]
    t, st, _ = _targets(spec, lead=5)
    assert t["when"].tolist() == [0, 1, -1, 0] and st["play_moved"] == 1
    # not in hand at the lead row either way
    spec = [(0, G, 0, 0, {"hand": (7, 8, 11, 10)}), (10, G), (13, P, 9, 77), (20, G)]
    t, st, _ = _targets(spec, lead=5)
    assert t["when"].tolist() == [0, 1, -1, 0] and st["play_moved"] == 1


def test_two_plays_in_one_interval_and_lost_plays_are_not_called_waits():
    spec = [(0, G), (10, G), (13, P, 9, 1), (16, P, 8, 2), (20, G), (30, G)]
    t, st, _ = _targets(spec, lead=0)
    assert t["when"].tolist() == [0, 1, -1, -1, 0, 0]       # the second has no free row <= 16
    assert st["play_lost_taken"] == 1 and st["play_assigned"] == 1
    # with a lead the first takes the earlier row and both are kept
    t, st, _ = _targets(spec, lead=5)
    assert t["when"].tolist() == [1, 1, -1, -1, 0, 0] and st["play_lost_taken"] == 0
    assert t["w_slot"][[0, 1]].tolist() == [2, 1]
    # a play that fits nowhere (card never in hand): its natural row is ignored, not a wait
    t, st, _ = _targets([(0, G), (10, G), (13, P, 55, 1), (20, G)], lead=0)
    assert t["when"].tolist() == [0, -1, -1, 0] and st["play_lost_infeasible"] == 1


def test_bent_plays_and_overdue_rows_are_ignored():
    spec = [(0, G), (10, G), (13, P, 9, 1, {"flags": D.F_REFUSED}), (20, G, 0, 0, {"flags": D.F_PENDING}), (30, G)]
    t, st, _ = _targets(spec, lead=0)
    assert t["when"].tolist() == [0, -1, -1, -1, 0] and st["play_bad"] == 1


def test_upcoming_play_fields():
    spec = [(0, G), (10, G), (13, P, 9, 77), (20, G), (26, P, 8, 5), (30, G)]
    t, _, _ = _targets(spec, lead=0)
    assert t["u_dt"].tolist() == [13, 3, 0, 6, 0, T.NO_DT]
    assert t["u_slot"].tolist() == [2, 2, 2, 1, 1, -1] and t["u_tile"].tolist() == [77, 77, 77, 5, 5, -1]
    # the next card is not in the hand slot yet on an earlier row: unusable there
    spec[0] = (0, G, 0, 0, {"hand": (7, 8, 11, 10)})
    t, _, _ = _targets(spec, lead=0)
    assert t["u_slot"].tolist()[:2] == [-1, 2]


def test_ability_targets_per_engine_slot():
    A = D.K_ABILITY
    spec = [(0, G, 0, 0, {"ab_ready": 0}), (10, G, 0, 0, {"ab_ready": 1}), (14, A, 0, 0, {"ability": 1, "ab_ready": 1}),
            (20, G, 0, 0, {"ab_ready": 2}), (30, G, 0, 0, {"ab_ready": 3})]
    t, st, _ = _targets(spec, lead=0)
    assert t["ab"][:, 0].tolist() == [-1, 1, -1, -1, 0]     # slot 1: fired from the row at tick 10
    assert t["ab"][:, 1].tolist() == [-1, -1, -1, 0, 0]     # slot 2: ready, never used
    assert st["ability_assigned"] == 1


def test_what_rows_selection():
    b = {"t_when": torch.tensor([1, 0, 0, 0, -1]), "t_w_slot": torch.tensor([2, -1, -1, -1, -1]),
         "t_w_tile": torch.tensor([9, -1, -1, -1, -1]), "t_u_slot": torch.tensor([1, 1, 3, -1, 0]),
         "t_u_tile": torch.tensor([4, 4, 6, -1, 8]), "t_u_dt": torch.tensor([3, 8, 0, 2, 12]),
         "kind": torch.tensor([G, G, P, G, G])}
    sel, slot, tile = T._what_rows(b, "none", True)
    assert sel.tolist() == [True, False, False, False, False] and (int(slot[0]), int(tile[0])) == (2, 9)
    sel, slot, _ = T._what_rows(b, "play", True)
    assert sel.tolist() == [True, False, True, False, False] and int(slot[2]) == 3
    sel, _, _ = T._what_rows(b, 5, True)
    assert sel.tolist() == [True, False, True, False, False]
    sel, _, _ = T._what_rows(b, 10, True)
    assert sel.tolist() == [True, True, True, False, False]
    sel, slot, _ = T._what_rows(b, 10, False)               # the ablation: next-play labels only
    assert sel.tolist() == [True, True, True, False, False] and int(slot[0]) == 1


class _DS:
    def __init__(self):
        self.info = {"fp_sha": "testfp"}
        self.metas = [{"name": "s0"}, {"name": "s1"}]
        self.root = "unused"


def test_noise_cache_scatters_the_cached_changes(tmp_path):
    nc = T.NoiseCache(_DS(), root=str(tmp_path))
    os.makedirs(nc.dir)
    # shard 0: row 0 unchanged, row 1 two entries, row 2 one entry; shard 1: row 0 one entry
    np.savez(nc.path(0), indptr=np.array([0, 0, 2, 3]), idx=np.array([1, 4, 0], np.uint16),
             val=np.array([0.5, -1.0, 7.0], np.float32))
    np.savez(nc.path(1), indptr=np.array([0, 1]), idx=np.array([5], np.uint16), val=np.array([9.0], np.float32))
    assert nc.ensure([0, 1], procs=1, log=None) == 0        # nothing to compute
    ref = torch.tensor([[0, 2, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0]])
    obs = nc(torch.zeros(4, 6), {"ref": ref})
    want = torch.zeros(4, 6)
    want[0, 0], want[1, 5], want[3, 1], want[3, 4] = 7.0, 9.0, 0.5, -1.0
    assert torch.equal(obs, want)
    assert "noise_bt-spells_s0" in nc.dir and "testfp" in nc.dir


def _game_obs(n, seed=0):
    deck = list(C.deck_ids())
    rng = np.random.default_rng(seed)
    g = C.CE(deck, deck, 3600)
    g.seed(seed + 1)
    out, hands = [], []
    for k in range(n):
        g.set_elixir_for_team(0, float(rng.uniform(0, 10)))
        o = np.array(g.get_observation_for_team(0), np.float32)
        out.append(np.concatenate([o, np.zeros(OV.TAIL_SIZE, np.float32)]))
        g.step(-1, 0.0, 0.0, 7)
    return torch.from_numpy(np.stack(out))


def _batch(n=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = _game_obs(n, seed)
    r = lambda lo, hi, *s: torch.randint(lo, hi, s or (n,), generator=g)   # noqa: E731
    b = {"obs": obs, "kind": torch.full((n,), G), "won": r(-1, 2), "dt_next": r(0, 200),
         "nx_slot": r(0, 4), "nx_card": r(1, 100), "hand": r(1, 100, n, 4),
         "t_when": r(-1, 2), "t_w_slot": r(0, 4), "t_w_tile": r(0, 34 * 18),
         "t_u_slot": r(-1, 4), "t_u_tile": r(0, 34 * 18), "t_u_dt": r(0, 30), "t_ab": r(-1, 2, n, 2)}
    b["t_w_slot"] = torch.where(b["t_when"] == 1, b["t_w_slot"], torch.full((n,), -1))
    return b


def test_losses_are_finite_and_every_part_trains_the_net():
    torch.manual_seed(0)
    net = N3.new_net_v3("cpu").train()
    b = _batch()
    total, parts = T.forward_losses(net, b, what_h=10)
    assert set(parts) == set(T.LOSS_W) and all(torch.isfinite(v) for v in parts.values())
    total.backward()
    for name in ("when_head", "what_head", "place_ctx", "dt_head", "next_head", "value_head", "tail_enc", "core"):
        g = [p.grad for p in getattr(net, name).parameters() if p.grad is not None]
        assert g and all(torch.isfinite(x).all() for x in g), name
        if name != "tail_enc":                      # an all-zero tail still trains its biases only
            assert any(float(x.abs().sum()) > 0 for x in g), name


def test_losses_with_nothing_to_learn_from_are_zero_not_nan():
    net = N3.new_net_v3("cpu").train()
    b = _batch(8)
    n = len(b["obs"])
    b["t_when"] = torch.full((n,), -1)
    b["t_u_slot"] = torch.full((n,), -1)
    b["t_ab"] = torch.full((n, 2), -1)
    b["nx_slot"] = torch.full((n,), 255)
    total, parts = T.forward_losses(net, b, what_h=10)
    assert torch.isfinite(total)
    assert float(parts["when"]) == float(parts["what"]) == float(parts["where"]) == float(parts["ability"]) == 0.0
    assert float(parts["next"]) == 0.0
    total.backward()


def test_value_target_is_the_match_result_as_stored():
    net = N3.new_net_v3("cpu").eval()
    b = _batch(8)
    with torch.no_grad():
        _, parts, d = T.forward_losses(net, b, detail=True)
        v = net.heads(net.extract_features_hires(b["obs"])[0], d["mask"])["value"].squeeze(1)
    assert set(b["won"].tolist()) <= {-1, 0, 1}                # bcdata2: loss / draw / win
    assert torch.allclose(parts["value"], ((v - b["won"].float()) ** 2).mean())


def test_what_loss_skips_rows_whose_target_slot_is_not_playable():
    net = N3.new_net_v3("cpu").eval()
    b = _batch(16, seed=3)
    mask = net.affordability_mask(b["obs"])[:, :4]
    assert (~mask).any() and mask.any()                    # the fixture has both kinds of row
    _, _, d = T.forward_losses(net, b, what_h=10, detail=True)
    slot = d["slot"].clamp(min=0)
    assert torch.equal(d["legal"], d["sel"] & mask.gather(1, slot.unsqueeze(1)).squeeze(1))


@pytest.mark.skipif(not os.path.exists(os.path.join(PILOT, "dataset.json")), reason="needs data/bc2/pilot")
def test_real_batches_targets_and_label_metrics():
    ds = D.BCDataset(PILOT)
    ld, tg, cache = T.make_loader(ds, "val", 0, "cpu", mirrored=False, noise=False, batch_size=512, shuffle=False)
    assert cache is None and tg.D == T.dataset_grid(ds)
    b = next(iter(ld))
    for k in ("t_when", "t_w_slot", "t_w_tile", "t_u_dt", "t_u_slot", "t_u_tile", "t_ab"):
        assert k in b and len(b[k]) == len(b["obs"])
    grid = (b["kind"] & D.K_GRID) != 0
    assert (b["t_when"][~grid] == -1).all() and (b["t_when"][grid] == 1).any()
    on = b["t_when"] == 1                                   # an assigned play names a card in hand at that row
    assert (b["t_w_slot"][on] >= 0).all() and (b["t_w_tile"][on] >= 0).all()
    net = N3.new_net_v3("cpu")
    m = T.label_metrics(net, ld, what_h=10, max_batches=2)
    assert m["rows"] == 1024 and m["when"]["rows"] > 0
    assert sum(r["n"] for r in m["when"]["reliability"]) == m["when"]["rows"]
    assert 0.0 <= m["what"]["top1_when_rows"] <= 1.0 and m["where"]["n"] > 0
    assert "The Log" in m["what"]["per_card"] or m["triggers"]["log_target"]["n"] == 0
    assert "WHEN" in T.format_label_metrics(m)


@pytest.mark.skipif(not os.path.exists(os.path.join(PILOT, "dataset.json")), reason="needs data/bc2/pilot")
def test_trainer_writes_a_checkpoint_load_policy_accepts(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "RUNS", str(tmp_path))
    a = argparse.Namespace(data=PILOT, name="t", lead=5, what_h="5", no_what_on_when=False, groups="1,2", epochs=1,
                           max_steps=2, batch_size=256, lr=1e-3, weight_decay=1e-4, warmup=2, clip=5.0, seed=0,
                           no_noise=True, noise_p=1.0, noise_seed=0, names="spells", no_tail=False, tail_dim=None,
                           core_hidden=None, core_layers=None, procs=1, device="cpu", log_every=1, val_batches=1)
    path = T.train(a, log=lambda *_: None)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ds = D.BCDataset(PILOT)
    assert ck["arch"] == "v3" and ck["lead"] == 5 and ck["obs_v2"] is True and ck["obs_dim"] == 14607
    assert ck["decision_ticks"] == T.dataset_grid(ds) and ck["dataset_fp"] == ds.info["fp_sha"]
    assert ck["dataset"] == "data/bc2/pilot" and ck["code_hash"] == N3.code_hash()
    assert ck["names"] == "all"                              # trained without tail noise: every name was shown
    assert ck["train"]["what_h"] == 5 and ck["split"] == "train"
    net, info = OC.load_policy(path)
    assert info["arch"] == "v3" and info["lead"] == 5 and info["decision_ticks"] == ck["decision_ticks"]
    ck["obs_dim"] += 1
    bad = str(tmp_path / "bad.pth")
    torch.save(ck, bad)
    with pytest.raises(ValueError):
        OC.load_policy(bad)


def test_training_never_reads_the_validation_split():
    import inspect

    src = inspect.getsource(T.train)
    assert 'RowFilter(split="train"' in src and 'ds.select("train"' in src
    assert 'split="val"' not in src.replace('make_loader(ds, "val"', "")   # val only through the metrics loader
