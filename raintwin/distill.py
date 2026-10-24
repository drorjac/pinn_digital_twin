"""Knowledge distillation: a small CNN learns to reproduce a teacher's rain maps.

The teacher is anything that makes a map for every 5-min step of the training period: the
multi-sensor twin (slow: an ensemble, a filter, every sensor), the radar, or a merged product.
The student sees only cheap inputs (here the link and station maps, at t, t-5 and t-10 min)
and outputs the teacher's map and, when the teacher has one, its spread. Trained in log space,
log(1 + R), so that light rain is not ignored next to heavy cells.
"""

import numpy as np
import torch
from torch import nn

INPUTS = ("C", "Ci", "P", "Pi")        # cube channels: link IDW map + coverage, station map + coverage
LAGS = (0, 1, 2)                       # steps back in time


def stack_inputs(cube, idx):
    """(n, len(INPUTS) * len(LAGS), ny, nx) float32 inputs for steps idx (log1p of rain maps)."""
    chans = []
    for lag in LAGS:
        j = np.maximum(np.asarray(idx) - lag, 0)
        for name in INPUTS:
            a = np.nan_to_num(np.asarray(cube._load(name, j), dtype="float32"))
            chans.append(np.log1p(np.clip(a, 0, None)) if name in ("C", "P") else a)
    return np.stack(chans, 1)


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GELU(),
                               nn.Conv2d(cout, cout, 3, padding=1), nn.GELU())

    def forward(self, x):
        return self.f(x)


class Student(nn.Module):
    """A two-level U-Net. Input is padded to a multiple of 4 and cropped back."""

    def __init__(self, cin, cout=2, width=32):
        super().__init__()
        self.e1, self.e2 = Block(cin, width), Block(width, 2 * width)
        self.mid = Block(2 * width, 2 * width)
        self.d2, self.d1 = Block(4 * width, width), Block(2 * width, width)
        self.out = nn.Conv2d(width, cout, 1)
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        H, W = x.shape[-2:]
        ph, pw = (-H) % 4, (-W) % 4
        x = nn.functional.pad(x, (0, pw, 0, ph), mode="replicate")
        a = self.e1(x)
        b = self.e2(self.pool(a))
        m = self.mid(self.pool(b))
        b = self.d2(torch.cat([self.up(m), b], 1))
        a = self.d1(torch.cat([self.up(b), a], 1))
        return self.out(a)[..., :H, :W]


def train(X, Y, M=None, Xv=None, Yv=None, Mv=None, epochs=25, batch=64, lr=2e-3, seed=0,
          width=32, log=print):
    """Fit a Student to targets Y (n, cout, ny, nx), already in log1p units. M: optional mask
    (n, 1, ny, nx) of cells that count in the loss (e.g. where the teacher is defined).
    Returns the model with the best validation loss."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Student(X.shape[1], Y.shape[1], width)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    Mt = torch.from_numpy(M) if M is not None else torch.ones_like(Yt[:, :1])

    def loss_on(xb, yb, mb):
        return (((net(xb) - yb) ** 2) * mb).sum() / (mb.sum() * yb.shape[1]).clamp(min=1)

    best, best_state, hist = np.inf, None, []
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(len(X))
        tot = 0.0
        for i in range(0, len(X), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            l = loss_on(Xt[b], Yt[b], Mt[b])
            l.backward()
            opt.step()
            tot += l.item() * len(b)
        sched.step()
        vl = np.nan
        if Xv is not None:
            vl = evaluate_loss(net, Xv, Yv, Mv)
            if vl < best:
                best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
        hist.append((tot / len(X), vl))
        log(f"epoch {ep + 1:2d}: train {tot / len(X):.4f}  validation {vl:.4f}")
    if best_state is not None:
        net.load_state_dict(best_state)
    net.history = np.array(hist)
    return net


@torch.no_grad()
def evaluate_loss(net, X, Y, M=None, batch=256):
    net.eval()
    num = den = 0.0
    for i in range(0, len(X), batch):
        xb, yb = torch.from_numpy(X[i:i + batch]), torch.from_numpy(Y[i:i + batch])
        mb = torch.from_numpy(M[i:i + batch]) if M is not None else torch.ones_like(yb[:, :1])
        num += (((net(xb) - yb) ** 2) * mb).sum().item()
        den += mb.sum().item() * yb.shape[1]
    return num / max(den, 1)


@torch.no_grad()
def predict(net, X, batch=256):
    """Rain maps (n, cout, ny, nx) in mm/h from log1p outputs."""
    net.eval()
    out = [net(torch.from_numpy(X[i:i + batch])).numpy() for i in range(0, len(X), batch)]
    return np.expm1(np.clip(np.concatenate(out), 0, None))
