"""A privileged residual critic for ppo_pool (opt-in: --priv-critic).

Asymmetric actor-critic: the ACTOR sees only what the live driver has (and,
with training/sim2real.py, a perturbed version of it); the value estimate used
for the advantages may also see the true state. Here that is

    V(s) = V_net(obs) + g(priv)

with V_net the policy net's own value head, exactly as before, and g a small
separate network on the scalar part of pool_env's privileged vector: the true
elixir of both sides, the opponent's hand and next card, and the six true tower
health fractions (pool_env.PRIV, the first 18 entries; the unit planes are left
out, they would cost 5,712 floats per step per env for what the board mostly
shows already).

Why a residual, outside the net:
  * the policy net, its inputs, its optimizer and its state dict are untouched,
    so a checkpoint trained with the flag plays live like any other (the phone
    has no privileged input) and old checkpoints resume with the flag on;
  * g's last layer starts at zero, so the first update with the flag is the
    update without it, and the flag can be switched on in a running lineage;
  * V_net still regresses on the return, so it stays a usable critic (the
    explained-variance line, bootstrapping after a resume without the flag).

g is trained on return - V_net(obs) (V_net detached) with its own Adam; no
gradient reaches the net. State: state.pth["priv_critic"].
"""
import numpy as np
import torch
from torch import nn

from training import pool_env as PE

N_IN = int(PE.PRIV["units"].start)        # the scalars before the unit planes


class PrivCritic:
    def __init__(self, n_envs, lr=3e-4, hidden=64, device="cpu"):
        self.n = int(n_envs)
        self.device = torch.device(device)
        # nn.Linear draws its default weights from the process-wide generator,
        # which the rollout samples actions from: put the stream back.
        keep = torch.get_rng_state()
        self.g = nn.Sequential(nn.Linear(N_IN, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                               nn.Linear(hidden, 1))
        torch.set_rng_state(keep)
        self.g.to(self.device)
        gen = torch.Generator().manual_seed(77017)          # our own weights, our own stream
        with torch.no_grad():
            for m in self.g:
                if isinstance(m, nn.Linear):
                    bound = 1.0 / np.sqrt(m.in_features)
                    m.weight.copy_((torch.rand(m.weight.shape, generator=gen) * 2 - 1) * bound)
                    m.bias.zero_()
            self.g[-1].weight.zero_()
        self.opt = torch.optim.Adam(self.g.parameters(), lr=lr)
        self.cur = None            # (n, N_IN): the state the next action is chosen in
        self.rows = []
        self._batch = None
        self.losses = []

    # -- rollout -------------------------------------------------------------------
    @staticmethod
    def _cut(priv):
        return np.stack([np.asarray(p, dtype=np.float32)[:N_IN] for p in priv])

    def start(self, reset_infos):
        """The privileged state of the observation the trainer holds after reset."""
        if reset_infos is None or reset_infos.get("priv") is None:
            raise RuntimeError("--priv-critic needs the reset info's 'priv' (pool: \"priv\": true)")
        self.cur = self._cut(reset_infos["priv"])

    def record(self, infos):
        """Once per env step, after it: file the state the action was chosen in,
        keep the one the step led to (info is post-step, i.e. the next row's)."""
        self.rows.append(self.cur)
        priv = infos.get("priv")
        if priv is None:
            raise RuntimeError("--priv-critic needs info['priv'] (pool: \"priv\": true)")
        self.cur = self._cut(priv)

    # -- update --------------------------------------------------------------------
    @torch.no_grad()
    def residuals(self, steps):
        """-> (T, n) residual for the stored rows, (n,) for the state after them."""
        if len(self.rows) != int(steps):
            raise RuntimeError(f"privileged rows {len(self.rows)} != rollout steps {steps}")
        self._batch = torch.as_tensor(np.stack(self.rows), dtype=torch.float32, device=self.device)
        self.rows = []
        self.losses = []
        nxt = torch.as_tensor(self.cur, dtype=torch.float32, device=self.device)
        return self.g(self._batch).squeeze(-1), self.g(nxt).squeeze(-1)

    def train_step(self, loc):
        """Inside the PPO minibatch (the updater's locals): one step of g on
        return - V_net. Touches nothing of the policy net's graph."""
        x = self._batch[loc["tt"], loc["ee"]]
        target = (loc["mb_ret"] - loc["new_values"].detach().reshape(loc["mb_ret"].shape)).detach()
        valid = loc["mb_valid"]
        with torch.enable_grad():
            pred = self.g(x).squeeze(-1)
            loss = 0.5 * (((pred - target) ** 2) * valid).sum() / valid.sum().clamp(min=1.0)
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.g.parameters(), 1.0)
            self.opt.step()
        self.losses.append(float(loss.detach()))

    def done(self):
        self._batch = None

    # -- state ---------------------------------------------------------------------
    def state(self):
        return {"g": {k: v.detach().cpu().clone() for k, v in self.g.state_dict().items()},
                "opt": self.opt.state_dict(), "n_in": N_IN}

    def load(self, blob):
        if not blob:
            return False
        self.g.load_state_dict(blob["g"])
        self.opt.load_state_dict(blob["opt"])
        return True
