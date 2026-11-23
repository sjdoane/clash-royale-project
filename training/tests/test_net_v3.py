"""NetV3 (training/net_v3.py): the factorised action, the tail, the interface the
PPO trainer / eval_ghost / live driver use, and checkpoint metadata.

    CR_ENGINE=build/gen2/engine .venv/bin/python -m pytest training/tests/test_net_v3.py -q
"""
import copy
import os

import numpy as np
import pytest
import torch

from training import common as C  # noqa: F401  (sets CLASH_DECK first)
from training import obs_compat as OC

pytestmark = pytest.mark.skipif(not OC.IS_GEN2, reason="NetV3 reads the gen-2 observation")
if OC.IS_GEN2:
    from training import net_v3 as N3
    from training import obs_v2 as OV

META = dict(names="spells", decision_ticks=5, lead=0, dataset="data/bc2/x", dataset_fp="abc")


def _obs(n=6, seed=0, tail=True):
    """Real engine observations (a seeded mirror match, both elixirs topped up),
    with a hand-written tail: one opponent play, one own play."""
    deck = list(C.deck_ids())
    rng = np.random.default_rng(seed)
    out = []
    g = C.CE(deck, deck, 3600)
    g.seed(seed + 1)
    for k in range(n):
        g.set_elixir_for_team(0, float(rng.uniform(0, 10)) if k else 10.0)
        o = np.array(g.get_observation_for_team(0), np.float32)
        if tail:
            t = np.zeros(OV.TAIL_SIZE, np.float32)
            t[OV.PRESENT] = 1.0
            e = t[OV.OPP_PLAYS].reshape(OV.K_OPP, OV.ENTRY)
            e[0] = [1, 1, (33 + 1) / 256, 0.2, 1, 0, 1, rng.integers(0, 18) / 17, rng.integers(0, 34) / 33, 0.05 * k]
            t[OV.OPP_PLAYS] = e.reshape(-1)
            t[OV.SCALARS] = [0.05 * k, 0.1, 0.0, 1.0]
            o = np.concatenate([o, t])
        out.append(o)
        g.step(-1, 0.0, 0.0, 7)
    return torch.from_numpy(np.stack(out))


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    n = N3.new_net_v3("cpu")
    with torch.no_grad():                      # untrained heads are near zero: spread them
        for p in n.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return n.eval()


def _step(net, obs):
    mask = net.affordability_mask(obs)
    f, emb, sp, hi = net.extract_features_hires(obs)
    h0 = (torch.zeros(len(obs), net.LSTM_HIDDEN), torch.zeros(len(obs), net.LSTM_HIDDEN))
    return (mask, f, emb, sp, hi) + tuple(net.step_lstm_and_card(f, h0, mask))


def test_shapes_and_observation_length(net):
    obs = _obs()
    assert obs.shape[1] == net.obs_dim == N3.engine_obs_dim() + OV.TAIL_SIZE == 14607
    mask, f, emb, sp, hi, cl, a1, a2, v, (hx, cx) = _step(net, obs)
    assert f.shape == (6, net.lstm_input_dim) and emb.shape == (6, 5, 16)
    assert cl.shape == (6, 5) and v.shape == (6, 1) and hx.shape == cx.shape == (6, net.LSTM_HIDDEN)
    pl = net.placement_given_card(hx, emb, torch.zeros(6, dtype=torch.long), obs, sp, hires_map=hi)
    assert pl.shape == (6, net.placement_cells) and torch.isfinite(pl).any(1).all()


def test_card_logits_compose_when_and_what(net):
    obs = _obs()
    mask, f, *_rest = _step(net, obs)
    cl = _rest[3]
    o = net.heads(f, mask)
    assert torch.allclose(torch.logsumexp(cl, -1), torch.zeros(6), atol=1e-5)      # normalised
    p_when = torch.sigmoid(o["when"])
    legal = o["has_legal"]
    assert legal.any()
    assert torch.allclose(cl[legal, 4].exp(), 1 - p_when[legal], atol=1e-5)
    what = torch.softmax(o["what"], -1)
    assert torch.allclose(cl[legal, :4].exp(), p_when[legal, None] * what[legal], atol=1e-5)
    assert (cl[:, :4][~mask[:, :4]] == float("-inf")).all()                        # unaffordable


def test_nothing_playable_is_a_certain_wait(net):
    obs = _obs()
    f = net.extract_features_hires(obs)[0]
    mask = torch.zeros(6, 5, dtype=torch.bool)
    mask[:, 4] = True
    cl = net.step_lstm_and_card(f, None, mask)[0]
    assert (cl[:, 4] == 0).all() and (cl[:, :4] == float("-inf")).all()
    assert torch.isfinite(torch.distributions.Categorical(logits=cl).entropy()).all()


def test_card_temperature_leaves_when_alone(net):
    obs = _obs()
    base = _step(net, obs)[5]
    try:
        net.set_sampling(what_temp=0.25, where_temp=0.25)
        cold = _step(net, obs)
        assert torch.allclose(cold[5][:, 4], base[:, 4], atol=1e-6)                # p(wait) unchanged
        m = torch.isfinite(base[:, :4]).sum(1) >= 2
        assert m.any() and not torch.allclose(cold[5][m, :4], base[m, :4], atol=1e-4)
        mask, f, emb, sp, hi = cold[:5]
        hx = cold[9][0]
        ci = torch.zeros(6, dtype=torch.long)
        p_cold = net.placement_given_card(hx, emb, ci, obs, sp, hires_map=hi)
        net.set_sampling()
        p_base = net.placement_given_card(hx, emb, ci, obs, sp, hires_map=hi)
        fin = torch.isfinite(p_base)
        assert torch.allclose(p_cold[fin], p_base[fin] / 0.25, atol=1e-4)
        net.set_sampling(when_bias=1.5)
        o = net.heads(f, mask)
        biased = _step(net, obs)[5]
        legal = o["has_legal"]
        assert torch.allclose(biased[legal, 4].exp(), 1 - torch.sigmoid(o["when"][legal] + 1.5), atol=1e-5)
    finally:
        net.set_sampling()
    with pytest.raises(ValueError):
        net.set_sampling(what_temp=0.0)


def test_internal_when_coin_makes_a_caller_temperature_harmless(net):
    obs = _obs(1).repeat(4000, 1)
    mask = net.affordability_mask(obs)
    f = net.extract_features_hires(obs)[0]
    p_when = torch.sigmoid(net.heads(f[:1], mask[:1])["when"])[0].item()
    try:
        net.set_sampling(when_internal=True)
        torch.manual_seed(3)
        cl = net.step_lstm_and_card(f, None, mask)[0]
    finally:
        net.set_sampling()
    waits = cl[:, 4] == 0
    plays = cl[:, 4] == float("-inf")
    assert (waits | plays).all()
    assert (cl[waits, :4] == float("-inf")).all()
    assert abs(plays.float().mean().item() - p_when) < 0.03
    # a caller's temperature on these logits cannot turn a play into a wait
    cold = torch.distributions.Categorical(logits=cl / 0.5).sample()
    assert ((cold == 4) == waits).all()


def test_the_tail_is_read_and_can_be_switched_off(net):
    obs = _obs()
    a = _step(net, obs)[5]
    other = obs.clone()
    other[:, net.base_obs_dim + OV.OPP_PLAYS.start + OV.E_CODE] = (7 + 1) / 256      # another card
    b = _step(net, other)[5]
    assert not torch.allclose(a[torch.isfinite(a)], b[torch.isfinite(b)], atol=1e-5)
    blind = N3.new_net_v3("cpu", use_tail=False).eval()
    sd = {k: v for k, v in net.state_dict().items()}
    blind.load_state_dict(sd)                                                        # same shapes
    a0, b0 = _step(blind, obs)[5], _step(blind, other)[5]
    assert torch.equal(a0, b0)
    # an observation without a tail is accepted and reads as "no tail"
    short = _step(blind, obs[:, :net.base_obs_dim])[5]
    assert torch.equal(short, a0)
    assert torch.equal(_step(net, obs[:, :net.base_obs_dim])[5], a0)


def test_embedding_table_and_padding(net):
    emb = net.tail_enc.embed
    assert emb.num_embeddings == 256 and emb.padding_idx == 0
    assert N3.DEFAULTS["tail_embed"] == emb.embedding_dim


def test_cycle_branch_and_tail_feed_the_actor_with_gradient(net):
    obs = _obs()
    n = copy.deepcopy(net).train()
    c0 = n.spatial_size + n.cycle_start            # a fresh match has an empty cycle block: fill it
    obs[:, c0:c0 + n.cycle_block_size] = torch.rand(len(obs), n.cycle_block_size)
    mask = n.affordability_mask(obs)
    f = n.extract_features_hires(obs)[0]
    o = n.heads(f, mask)
    (o["when"].sum() + o["what"][torch.isfinite(o["what"])].sum()).backward()
    assert n.scalar_mlp.cycle_out.weight.grad.abs().sum() > 0
    assert n.tail_enc.embed.weight.grad.abs().sum() > 0
    assert n.tail_enc.embed.weight.grad[0].abs().sum() == 0                          # padding row


def test_hidden_state_is_ignored(net):
    obs = _obs()
    mask = net.affordability_mask(obs)
    f = net.extract_features_hires(obs)[0]
    a = net.step_lstm_and_card(f, (torch.zeros(6, 256), torch.zeros(6, 256)), mask)
    b = net.step_lstm_and_card(f, (torch.randn(6, 256), torch.randn(6, 256)), mask)
    assert torch.equal(a[0], b[0]) and torch.equal(a[4][0], b[4][0]) and (a[4][1] == 0).all()


def test_ability_heads_are_generic_per_engine_slot(net):
    obs = _obs()
    f = net.extract_features_hires(obs)[0]
    n = copy.deepcopy(net)
    shapes = {k: tuple(v.shape) for k, v in n.state_dict().items()}
    outs = {}
    for slots in ([], [1], [2], [1, 2]):
        n.set_ability_slots(slots)
        assert n.num_ability_slots == len(slots)
        _, a1, a2, _, _ = n.step_lstm_and_card(f, None, None)
        assert len([x for x in (a1, a2) if x is not None]) == len(slots)
        outs[tuple(slots)] = (a1, a2)
        assert {k: tuple(v.shape) for k, v in n.state_dict().items()} == shapes      # weights do not move
    assert torch.equal(outs[(1,)][0], outs[(1, 2)][0]) and torch.equal(outs[(2,)][0], outs[(1, 2)][1])
    assert not torch.equal(outs[(1,)][0], outs[(2,)][0])
    with pytest.raises(ValueError):
        n.set_ability_slots([0])
    assert not any("deck" in k for k in shapes)


def test_forward_sequence_matches_the_step_interface(net):
    obs = _obs(8)
    mask, f, emb, sp, hi, cl, a1, a2, v, (hx, _) = _step(net, obs)
    torch.manual_seed(0)
    ci = torch.distributions.Categorical(logits=cl).sample()
    ci[0], ci[1] = 4, 0 if mask[1, 0] else 4
    pl = net.placement_given_card(hx, emb, ci, obs, sp, hires_map=hi)
    L, B = 4, 2
    sh = lambda t: t.view(L, B, *t.shape[1:])   # noqa: E731
    active = torch.nonzero(ci != 4).squeeze(1)
    out = net.forward_sequence(sh(f), sh(emb), sh(sp), sh(obs), sh(mask), sh(ci), torch.ones(L, B),
                               (torch.zeros(B, 256), torch.zeros(B, 256)), hires_seq=sh(hi),
                               active_rows=active, with_ability=True)
    card, place, values, aux, (h_last, c_last), extra, ab = out
    assert card.shape == (L, B, 5) and place.shape == (L, B, 612) and values.shape == (L, B)
    assert aux.shape == (L, B, int(C.CE.NUM_CARD_IDS)) and extra is None
    fin = torch.isfinite(cl)
    assert torch.allclose(card.view(-1, 5)[fin], cl[fin], atol=1e-5)
    assert torch.allclose(values.view(-1), v.view(-1), atol=1e-5)
    got, want = place.view(-1, 612)[active], pl[active]
    fin = torch.isfinite(want)
    assert (torch.isfinite(got) == fin).all() and torch.allclose(got[fin], want[fin], atol=1e-4)
    idle = torch.nonzero(ci == 4).squeeze(1)
    assert (place.view(-1, 612)[idle] == 0).all()                                    # zero fill, not -inf
    assert len(ab) == net.num_ability_slots and all(x.shape == (L, B, 2) for x in ab)
    assert torch.allclose(h_last, hx.view(L, B, -1)[-1], atol=1e-5) and (c_last == 0).all()
    # the sampling knobs do not leak into the training-time log-probabilities
    try:
        net.set_sampling(what_temp=0.3, where_temp=0.3, when_bias=2.0)
        again = net.forward_sequence(sh(f), sh(emb), sh(sp), sh(obs), sh(mask), sh(ci), torch.ones(L, B),
                                     (torch.zeros(B, 256), torch.zeros(B, 256)), hires_seq=sh(hi))
    finally:
        net.set_sampling()
    fin = torch.isfinite(card)
    assert torch.equal(again[0][fin], card[fin])
    # forward_from_features = the two halves
    c2, p2, v2, *_ = net.forward_from_features(f, emb, None, ci, mask, obs, sp)
    fin = torch.isfinite(pl)
    assert torch.allclose(p2[fin], pl[fin], atol=1e-4) and torch.equal(c2, cl)


def test_size_is_near_the_small_net():
    n = N3.new_net_v3("cpu")
    total = N3.param_table(n)["total"]
    small = sum(p.numel() for p in C.new_net("cpu").parameters())
    assert 0.4 * small < total < 1.5 * small
    big = N3.param_table(N3.new_net_v3("cpu", core_hidden=1024, core_layers=2))["total"]
    assert big > 2 * total
    with pytest.raises(ValueError):
        N3.new_net_v3("cpu", trunk=3)


def test_dt_buckets():
    dt = torch.tensor([0, 4, 5, 9, 10, 19, 20, 39, 40, 79, 80, 5000, 65535])
    assert N3.dt_bucket(dt).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6]
    assert N3.N_DT == 7


def test_checkpoint_metadata_roundtrip_and_loud_mismatch(net, tmp_path):
    path = str(tmp_path / "v3.pth")
    with pytest.raises(ValueError):
        N3.save_v3(net, path, names="spells")                                        # metadata is required
    blob = N3.save_v3(net, path, **META)
    assert blob["arch"] == "v3" and blob["obs_dim"] == 14607 and blob["obs_v2"] is True
    assert blob["code_hash"] == N3.code_hash() and blob["decision_ticks"] == 5 and blob["lead"] == 0
    pol, info = OC.load_policy(path)
    assert isinstance(pol, N3.NetV3) and not pol.training
    assert info["arch"] == "v3" and info["decision_ticks"] == 5 and info["names"] == "spells"
    assert info["obs_v2"] is True and info["dataset_fp"] == "abc" and info["lead"] == 0
    assert pol.when_internal is True                                                 # safe for tempering callers
    obs = _obs()
    pol.set_sampling()
    assert torch.equal(_step(pol, obs)[5], _step(net, obs)[5])

    def broken(**change):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        for k, v in change.items():
            if v is None:
                ck.pop(k)
            else:
                ck[k] = v
        p = str(tmp_path / "bad.pth")
        torch.save(ck, p)
        return p

    for change in (dict(code_hash="0" * 16), dict(obs_dim=14459), dict(base_obs_dim=13000), dict(obs_v2=False),
                   dict(num_card_ids=140), dict(lead=None), dict(decision_ticks=None), dict(names=None),
                   dict(dataset_fp=None), dict(cfg=dict(N3.DEFAULTS, core_hidden=64))):
        with pytest.raises(ValueError):
            OC.load_policy(broken(**change))
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ck["model"].pop("when_head.weight")
    torch.save(ck, str(tmp_path / "bad2.pth"))
    with pytest.raises(ValueError):
        OC.load_policy(str(tmp_path / "bad2.pth"))


def test_code_hash_ignores_comments_only(tmp_path):
    src = open(N3.__file__).read()
    a = tmp_path / "a.py"
    a.write_text(src + "\n# a trailing comment\n")
    b = tmp_path / "b.py"
    b.write_text(src + "\nX = 1\n")
    assert N3.code_hash(str(a)) == N3.code_hash() != N3.code_hash(str(b))


def test_old_checkpoints_still_load():
    p = os.path.join(C.REPO, "remote_results", "bc_hog26.pth")
    if not os.path.exists(p):
        pytest.skip("no gen-1 checkpoint here")
    pol, info = OC.load_policy(p)
    assert info.get("arch") != "v3"
