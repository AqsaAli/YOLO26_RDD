import torch
import torch.nn as nn


# ── 1. GSConv ──────────────────────────────────────────
# Source: https://github.com/AlanLi1997/slim-neck-by-gsconv
# Paper: "Slim-neck by GSConv: A better design paradigm of detector architectures"

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Mish() if act else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GSConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, 2, c_, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        y = y.permute(0, 2, 1, 3, 4)
        return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])


# ── 2. DySample (Phase 2) ──────────────────────────────
# Source: https://github.com/tiny-smart/dysample
# Paper: "Learning to Upsample by Learning to Sample" (ICCV 2023)
# TODO: paste DySample class here in Phase 2


# ── 3. BiFPN_Concat (Phase 3) ──────────────────────────
# Source: EfficientDet (Tan et al., CVPR 2020)
# TODO: paste BiFPN_Concat class here in Phase 3
        