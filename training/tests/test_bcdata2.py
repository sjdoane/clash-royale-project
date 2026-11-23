"""Sharded BC storage and loader (training/bcdata2.py). No engine, no data:
every test writes its own small dataset.

    .venv/bin/python -m pytest training/tests/test_bcdata2.py -q -p no:cacheprovider
"""
import json
import os

import numpy as np
import pytest
import torch

from training import bcdata2 as D

DIM = 300
ALL = dict(split=None, kinds=D.K_PLAY | D.K_ABILITY | D.K_GRID | D.K_AUX, legal="any", faithful=False, t0=True)


def _episode(rng, n, deck_id, group, mirrored, split, m):
    lens = rng.integers(0, 12, n)
    idx = np.concatenate([np.sort(rng.choice(DIM, int(k), replace=False)) for k in lens] + [np.zeros(0, np.int64)])
    val = rng.uniform(0.1, 1.0, int(lens.sum())).astype(np.float32)
    rows = np.zeros(n, D.ROW)
    play = rng.random(n) < 0.35
    rows["tick"] = np.arange(n) * 10
    rows["skip"] = 10
    rows["kind"] = D.K_GRID | np.where(play, D.K_PLAY, 0)
    rows["slot"] = np.where(play, rng.integers(0, 4, n), D.WAIT)
    rows["tile"] = np.where(play, rng.integers(0, 18 * 17, n), -1)
    rows["card"] = np.where(play, rng.integers(1, 100, n), -1)
    rows["hand"] = rng.integers(1, 100, (n, 4))
    rows["ability"] = rng.integers(0, 4, n)
    rows["ab_ready"] = rng.integers(0, 4, n)
    rows["elixir"] = rng.uniform(0, 10, n)
    rows["dt_next"] = rng.integers(0, 300, n)
    rows["nx_slot"], rows["nx_tile"], rows["nx_card"] = rng.integers(0, 4, n), rng.integers(0, 300, n), rng.integers(1, 100, n)
    f = np.zeros(n, np.int64)
    f[0] |= D.F_T0
    f[n - 2:] |= D.F_PAST_FAITH | D.F_PAST_DIV
    f[rng.random(n) < 0.15] |= D.F_LATE
    f[rng.random(n) < 0.10] |= D.F_TOPUP
    f[(rng.random(n) < 0.10) & ~play] |= D.F_PENDING
    rows["flags"] = f
    ep = np.zeros(1, D.EP)
    ep["m"], ep["side"], ep["deck_id"], ep["opp_deck_id"] = m, m % 2, deck_id, deck_id + 7
    ep["group"], ep["mirrored"], ep["split"], ep["result"], ep["tags"] = group, mirrored, split, (-1, 0, 1)[m % 3], D.T_FREQ
    return ep, rows, lens, idx.astype(np.uint16), val


def _dense(lens, idx, val):
    out = np.zeros((len(lens), DIM), np.float32)
    a = 0
    for i, k in enumerate(lens):
        out[i, idx[a:a + k].astype(np.int64)] = val[a:a + k]
        a += int(k)
    return out


def make(root, values="float32", seed=0):
    """-> truth: shard name -> (dense obs, rows, per-row EP records)."""
    os.makedirs(os.path.join(root, "shards"), exist_ok=True)
    with open(os.path.join(root, "dataset.json"), "w") as f:
        json.dump({"format": D.FORMAT, "obs_dim": DIM, "values": values}, f)
    rng = np.random.default_rng(seed)
    truth, m = {}, 0
    for name, group, n_eps in (("train-oth-0000", 0, 4), ("train-oth-0001", 0, 3), ("train-hog-0000", 2, 3),
                               ("train-hog-0000-m", 2, 3), ("val-oth-0000", 0, 2)):
        w = D.ShardWriter(root, name, DIM, values)
        dense, rows, eps = [], [], []
        for _ in range(n_eps):
            ep, r, lens, idx, val = _episode(rng, int(rng.integers(9, 30)), deck_id=m % 5, group=group,
                                             mirrored=int(name.endswith("-m")), split=int(name.startswith("val")), m=m)
            w.add(ep, r, lens, idx, val, np.zeros(0, D.OPP))
            dense.append(_dense(lens, idx, val))
            rows.append(r)
            eps.append(np.repeat(ep, len(r)))
            m += 1
        w.close()
        truth[name] = (np.concatenate(dense), np.concatenate(rows), np.concatenate(eps))
    return truth


@pytest.fixture()
def data(tmp_path):
    root = str(tmp_path / "ds")
    return root, make(root)


def _collect(loader):
    out = []
    for b in loader:
        out.append({k: (v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in b.items()})
    return out


def test_shard_round_trip(data):
    root, truth = data
    ds = D.BCDataset(root, expect_obs_dim=DIM)
    assert len(ds) == 5 and ds.n_rows == sum(len(t[1]) for t in truth.values())
    for i, meta in enumerate(ds.metas):
        s = ds.open(i)
        dense, rows, eps = truth[meta["name"]]
        assert np.array_equal(s.dense(np.arange(len(rows))), dense)
        for k in D.ROW.names:
            if k != "ep":
                assert np.array_equal(s.rows[k], rows[k]), k
        assert np.array_equal(s.eps[s.rows["ep"]]["m"], eps["m"])
        for e in range(len(s.eps)):
            ep, sl = s.episode(e)
            assert (s.rows["ep"][sl] == e).all() and sl.stop - sl.start == ep["nrows"]
        assert meta["n_rows"] == len(rows) and meta["nnz"] == int((dense != 0).sum())


@pytest.mark.parametrize("batch,block", [(7, 2), (64, 1), (1000, 8)])
def test_loader_serves_every_stored_row_once(data, batch, block):
    root, truth = data
    ds = D.BCDataset(root)
    ld = ds.loader(batch_size=batch, flt=D.RowFilter(**ALL), block=block, seed=3)
    got = _collect(ld)
    names = [m["name"] for m in ds.metas]
    total = sum(len(t[1]) for t in truth.values())
    assert sum(len(b["slot"]) for b in got) == total == ld.rows_served
    assert all(len(b["slot"]) == batch for b in got[:-1])
    seen = set()
    for b in got:
        for j in range(len(b["slot"])):
            sh, row, ep = (int(v) for v in b["ref"][j])
            dense, rows, eps = truth[names[sh]]
            r, e = rows[row], eps[row]
            assert (sh, row) not in seen
            seen.add((sh, row))
            assert np.array_equal(b["obs"][j], dense[row])
            assert (b["slot"][j], b["tile"][j], b["card"][j]) == (r["slot"], r["tile"], r["card"])
            assert list(b["hand"][j]) == list(r["hand"])
            assert list(b["ability"][j]) == [r["ability"] & 1, r["ability"] >> 1 & 1]
            assert list(b["ab_ready"][j]) == [r["ab_ready"] & 1, r["ab_ready"] >> 1 & 1]
            assert (b["kind"][j], b["flags"][j], b["tick"][j], b["dt_next"][j]) == (r["kind"], r["flags"], r["tick"], r["dt_next"])
            assert (b["nx_slot"][j], b["nx_tile"][j], b["nx_card"][j]) == (r["nx_slot"], r["nx_tile"], r["nx_card"])
            assert b["elixir"][j] == r["elixir"]
            assert (b["deck_id"][j], b["opp_deck_id"][j], b["won"][j], b["m"][j], b["side"][j]) == \
                (e["deck_id"], e["opp_deck_id"], e["result"], e["m"], e["side"])
            assert (b["mirrored"][j], b["group"][j]) == (e["mirrored"], e["group"])
    assert len(seen) == total


def test_shuffle_is_seeded_and_can_be_off(data):
    root, truth = data
    ds = D.BCDataset(root)
    refs = lambda **kw: np.concatenate([b["ref"] for b in _collect(ds.loader(batch_size=16, flt=D.RowFilter(**ALL), **kw))])   # noqa: E731
    a, b, c = refs(seed=1), refs(seed=1), refs(seed=2)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    plain = refs(shuffle=False)
    assert np.array_equal(plain[:, 0], np.sort(plain[:, 0]))
    first = plain[plain[:, 0] == 0][:, 1]
    assert np.array_equal(first, np.arange(len(first)))
    two = np.concatenate([b["ref"] for b in _collect(ds.loader(batch_size=16, flt=D.RowFilter(**ALL), epochs=2))])
    assert len(two) == 2 * len(a)
    full = _collect(ds.loader(batch_size=16, flt=D.RowFilter(**ALL), drop_last=True))
    assert all(len(b["slot"]) == 16 for b in full) and sum(len(b["slot"]) for b in full) == len(a) // 16 * 16


def _no_t0(rows):
    r = rows.copy()
    r["flags"] &= ~np.uint16(D.F_T0)
    return r


def _count(truth, pred, names):
    return sum(int(pred(truth[n][1], truth[n][2]).sum()) for n in names)


def test_filters(data):
    root, truth = data
    ds = D.BCDataset(root)
    n_of = lambda **kw: sum(len(b["slot"]) for b in _collect(ds.loader(batch_size=50, flt=D.RowFilter(**kw))))   # noqa: E731
    train = [n for n in truth if n.startswith("train")]
    play = lambda r: (r["kind"] & D.K_PLAY) != 0   # noqa: E731

    def default(r, e):
        ok = (r["flags"] & (D.F_T0 | D.F_PAST_FAITH)) == 0
        ok &= ~(play(r) & ((r["flags"] & D.BAD_PLAY) != 0))
        return ok & ~(~play(r) & ((r["flags"] & D.F_PENDING) != 0))

    assert n_of() == _count(truth, default, train) > 0
    assert n_of(legal="exact") == _count(truth, lambda r, e: default(r, e) & ((r["flags"] & D.F_LATE) == 0), train)
    assert n_of(legal="exact") < n_of() < n_of(legal="any") < n_of(legal="any", faithful=False)
    assert n_of(split="val") == _count(truth, default, ["val-oth-0000"])
    assert n_of(split=None) == _count(truth, default, list(truth))
    assert n_of(mirrored=True) == _count(truth, default, ["train-hog-0000-m"])
    assert n_of(mirrored=False) == _count(truth, default, [n for n in train if not n.endswith("-m")])
    assert n_of(groups=(0,)) == _count(truth, default, ["train-oth-0000", "train-oth-0001"])
    assert n_of(groups=(1, 2), mirrored=False) == _count(truth, default, ["train-hog-0000"])
    assert n_of(decks=[1, 3]) == _count(truth, lambda r, e: default(r, e) & np.isin(e["deck_id"], [1, 3]), train)
    assert n_of(wait_keep=0.0) == _count(truth, lambda r, e: default(r, e) & play(r), train)
    assert n_of(kinds=D.K_PLAY) == n_of(wait_keep=0.0)
    assert n_of(undiverged=True) == _count(truth, lambda r, e: default(r, e) & ((r["flags"] & D.F_PAST_DIV) == 0), train)
    with_t0 = lambda r, e: default(r, e) | (   # noqa: E731  (default, tick-0 rows allowed)
        default(_no_t0(r), e) & ((r["flags"] & D.F_T0) != 0))
    assert n_of(t0=True) == _count(truth, with_t0, train) > n_of()
    assert n_of(fn=lambda c: c["tick"] < 50) == _count(truth, lambda r, e: default(r, e) & (r["tick"] < 50), train)
    cards = [5, 6, 7, 8, 9, 10]
    assert n_of(cards=cards) == _count(truth, lambda r, e: default(r, e) & (~play(r) | np.isin(r["card"], cards)), train)
    with pytest.raises(ValueError):
        ds.loader(flt=D.RowFilter(split="val", mirrored=True))      # nothing matches
    half = n_of(wait_keep=0.5)
    assert n_of(wait_keep=0.0) < half < n_of()


def test_hooks_and_weights(data):
    root, _ = data
    ds = D.BCDataset(root)
    calls = []

    def relabel(b):
        calls.append("label")
        b["slot"] = torch.where(b["dt_next"] < 10, b["nx_slot"], b["slot"])
        return b

    def noise(obs, b):
        calls.append("noise")
        assert obs.shape == (len(b["slot"]), DIM)
        return obs + 1.0

    table = np.arange(6, dtype=np.float32) + 1.0
    plain = _collect(ds.loader(batch_size=32, shuffle=False))
    hooked = _collect(ds.loader(batch_size=32, shuffle=False, label_transform=relabel, noise=noise,
                                weight_fn=D.table_weights(table, "deck_id")))
    assert calls[:2] == ["label", "noise"] and len(calls) == 2 * len(hooked)
    for p, h in zip(plain, hooked):
        assert p["weight"] is None
        assert np.allclose(h["obs"], p["obs"] + 1.0)
        assert np.array_equal(h["slot"], np.where(p["dt_next"] < 10, p["nx_slot"], p["slot"]))
        assert np.array_equal(h["weight"], table[p["deck_id"]])


def test_float16_values_and_device(data, tmp_path):
    root16 = str(tmp_path / "f16")
    truth = make(root16, values="float16")
    ds = D.BCDataset(root16)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    names = [m["name"] for m in ds.metas]
    for b in ds.loader(batch_size=40, device=dev, flt=D.RowFilter(**ALL)):
        assert b.obs.dtype == torch.float32 and b.obs.device.type == dev and b.slot.dtype == torch.int64
        obs, ref = b.obs.cpu().numpy(), b.ref.cpu().numpy()
        for j in range(len(ref)):
            want = truth[names[ref[j, 0]]][0][ref[j, 1]].astype(np.float16).astype(np.float32)
            assert np.array_equal(obs[j], want)


def test_observation_length_mismatch_fails_loudly(data):
    root, _ = data
    with pytest.raises(D.ObsDimMismatch):
        D.BCDataset(root, expect_obs_dim=DIM + 148)
    w = D.ShardWriter(root, "train-oth-0009", DIM)
    with pytest.raises(D.ObsDimMismatch):
        w.add(np.zeros(1, D.EP), np.zeros(1, D.ROW), [1], np.array([DIM], np.uint16), np.ones(1, np.float32),
              np.zeros(0, D.OPP))
    mp = os.path.join(root, "shards", "train-oth-0001", "meta.json")
    with open(mp) as f:
        meta = json.load(f)
    meta["obs_dim"] = DIM - 1
    with open(mp, "w") as f:
        json.dump(meta, f)
    with pytest.raises(D.ObsDimMismatch):
        D.BCDataset(root)


def test_unfinished_shards_are_invisible_and_summary_counts(data):
    root, truth = data
    os.makedirs(os.path.join(root, "shards", ".tmp-train-oth-0005-1"))
    os.makedirs(os.path.join(root, "shards", "train-oth-0006"))          # no meta.json: not finished
    ds = D.BCDataset(root)
    assert len(ds) == 5
    s = D.summary(ds)
    rows = np.concatenate([t[1] for t in truth.values()])
    assert s["rows"] == len(rows) and s["eps"] == 15
    assert s["play"] == int(((rows["kind"] & D.K_PLAY) != 0).sum())
    assert s["past_faith"] == int(((rows["flags"] & D.F_PAST_FAITH) != 0).sum())
    assert s["served_default"] == sum(len(b["slot"]) for b in _collect(ds.loader(flt=D.RowFilter(split=None))))


# ------------------------------------------------------------- tower guard ---
def _drop(a, names):
    """The same records without `names` (a shard written before those fields)."""
    dt = np.dtype([(n, a.dtype[n]) for n in a.dtype.names if n not in names])
    out = np.zeros(a.shape, dt)
    for n in dt.names:
        out[n] = a[n]
    return out


def test_guard_fields_are_served_filtered_and_optional(data):
    """Rows carry the hit points the tower guard had given back so far (and a
    flag once it has intervened); a shard written before those fields existed
    still loads, with zeros."""
    root, truth = data
    assert D.ROW.names[-2:] == D.ROW_ADDED == ("g_own", "g_opp") and D.FCOLS[0] == "elixir"
    assert set(D.EP_ADDED) <= set(D.EP.names) and D.F_GUARD == 4096
    # one shard gets guard values, another is turned into the old format
    new, old = "train-hog-0000", "train-oth-0001"
    f = os.path.join(root, "shards", new, "rows.npy")
    rows = np.load(f)
    half = np.arange(len(rows)) >= len(rows) // 2
    rows["g_own"], rows["g_opp"] = np.where(half, 1500.0, 0.0), np.where(half, 250.5, 0.0)
    rows["flags"] |= np.where(half, D.F_GUARD, 0).astype(rows["flags"].dtype)
    np.save(f, rows)
    f = os.path.join(root, "shards", new, "eps.npy")
    eps = np.load(f)
    eps["guard"], eps["g_first"], eps["g_heals"], eps["g_hp"] = 1, 40, (3, 1), (1500.0, 250.5)
    eps["king_asleep"], eps["king_hit"] = (1, 0), (70, -1)
    np.save(f, eps)
    for name, added in (("rows", D.ROW_ADDED), ("eps", D.EP_ADDED)):
        f = os.path.join(root, "shards", old, name + ".npy")
        np.save(f, _drop(np.load(f), added))
    ds = D.BCDataset(root)
    i_old = [m["name"] for m in ds.metas].index(old)
    s = ds.open(i_old, obs=False)
    assert "g_own" not in s.rows.dtype.names and "guard" not in s.eps.dtype.names
    li, lf = D.labels(s, i_old)
    assert lf.shape == (len(s.rows), len(D.FCOLS)) and not lf[:, 1:].any()
    assert np.array_equal(lf[:, D.FC["elixir"]], s.rows["elixir"])

    def served(**kw):
        out = {}
        for b in _collect(ds.loader(batch_size=64, flt=D.RowFilter(**{**ALL, **kw}))):
            for k in range(len(b["slot"])):
                out[(int(b["ref"][k, 0]), int(b["ref"][k, 1]))] = (float(b["g_own"][k]), float(b["g_opp"][k]),
                                                                    int(b["flags"][k]))
        return out

    everything = served()
    assert len(everything) == sum(len(t[1]) for t in truth.values())
    i_new = [m["name"] for m in ds.metas].index(new)
    got = np.array([everything[(i_new, k)] for k in range(len(rows))])
    assert np.array_equal(got[:, 0], rows["g_own"]) and np.array_equal(got[:, 1], rows["g_opp"])
    assert all(v[:2] == (0.0, 0.0) for (sh, _), v in everything.items() if sh != i_new)    # old and unguarded: 0
    after, before = served(guarded=True), served(guarded=False)
    assert len(after) == int(half.sum()) and set(after) == {(i_new, k) for k in np.flatnonzero(half)}
    assert len(before) == len(everything) - len(after) and not set(before) & set(after)
    assert len(served(guard_hp=1750.5)) == len(everything)                    # own + other seat, inclusive
    assert served(guard_hp=1750.0) == before
    assert served(guard_hp=0.0) == before
    with pytest.raises(ValueError, match="guard_hp"):
        D.RowFilter(guard_hp=10.0).mask(li)                                  # needs the float labels
    sm = D.summary(ds)
    assert sm["rows_guarded"] == int(half.sum()) and sm["eps_guard"] == sm["eps_guard_intervened"] == len(eps)
    assert sm["guard_heals"] == 4 * len(eps) and sm["guard_hp_own"] == 1500.0 * len(eps)
    assert sm["sides_king_asleep"] == sm["sides_king_woke_wrongly"] == len(eps)
    assert sm["rows"] == len(everything) and sm["eps_guard_breach"] == 0
    # the default filter does not look at the guard at all
    plain = D.RowFilter(split=None)
    assert plain.guarded is None and plain.guard_hp is None
