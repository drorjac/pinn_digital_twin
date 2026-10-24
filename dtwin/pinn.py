"""Physics-informed neural networks (PyTorch)."""

import time

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    """Fully connected tanh network with input normalisation to [-1, 1]."""

    def __init__(self, n_in, n_out, width=64, depth=4, lb=None, ub=None):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            lin = nn.Linear(d, width)
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.Tanh()]
            d = width
        out = nn.Linear(d, n_out)
        nn.init.xavier_normal_(out.weight)
        nn.init.zeros_(out.bias)
        self.net = nn.Sequential(*layers, out)
        lb = torch.zeros(n_in) if lb is None else torch.as_tensor(lb, dtype=torch.float32)
        ub = torch.ones(n_in) if ub is None else torch.as_tensor(ub, dtype=torch.float32)
        self.register_buffer("lb", lb)
        self.register_buffer("ub", ub)

    def forward(self, z):
        return self.net(2.0 * (z - self.lb) / (self.ub - self.lb) - 1.0)


def grad(y, x):
    """dy/dx for each sample, keeping the graph so higher derivatives work."""
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def tensor(a, requires_grad=False):
    return torch.tensor(np.asarray(a), dtype=torch.float32, requires_grad=requires_grad)


def train(loss_fn, params, n_adam=3000, lr=1e-3, n_lbfgs=0, log_every=500, callback=None):
    """loss_fn() -> (total_loss, {name: float}). Adam, then optional L-BFGS polishing.

    params: a list of tensors, or a list of Adam param-group dicts (e.g. a larger lr for a
    physical parameter). L-BFGS always uses a single group.

    Returns a history dict of lists (one entry per iteration, plus 'phase').
    """
    hist = {"total": [], "phase": []}
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.1 ** (1 / max(n_adam, 1)))
    t0 = time.time()

    def record(total, parts, phase):
        hist["total"].append(float(total))
        hist["phase"].append(phase)
        for k, v in parts.items():
            hist.setdefault(k, []).append(v)
        if callback is not None:
            callback(hist)

    for it in range(n_adam):
        opt.zero_grad()
        total, parts = loss_fn()
        total.backward()
        opt.step()
        sched.step()
        record(total.item(), parts, "adam")
        if log_every and (it % log_every == 0 or it == n_adam - 1):
            print(f"adam {it:5d}  loss {total.item():.3e}  " +
                  "  ".join(f"{k} {v:.2e}" for k, v in parts.items()) + f"  ({time.time()-t0:.0f}s)")

    if n_lbfgs:
        flat = [p for g in params for p in g["params"]] if isinstance(params[0], dict) else params
        opt = torch.optim.LBFGS(flat, lr=1.0, max_iter=n_lbfgs, history_size=50,
                                tolerance_grad=1e-9, tolerance_change=1e-12,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            total, parts = loss_fn()
            total.backward()
            record(total.item(), parts, "lbfgs")
            return total

        opt.step(closure)
        print(f"lbfgs done  loss {hist['total'][-1]:.3e}  ({time.time()-t0:.0f}s)")
    return hist
