"""Backbone + block-causal branch mask + pointer head (after kev's model.py).

Differences from kev: Gemma 4 base, our delimiter layout, and a mask that
also respects the sliding-window layers so training sees what llama.cpp
computes at inference.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def branch_mask(segs: list[list[int]], device, window: int | None = None, dtype=torch.float32):
    """Additive [B,1,L,L]: query i may attend key j iff j <= i, j is real, and
    seg[j] == 0 (state) or seg[j] == seg[i]. With `window`, additionally
    i - j < window (sliding-window layers)."""
    L = max(len(s) for s in segs)
    s = torch.full((len(segs), L), -1, device=device)
    for b, seg in enumerate(segs):
        s[b, : len(seg)] = torch.tensor(seg, device=device)
    i = torch.arange(L, device=device)
    causal = i[None, :] <= i[:, None]
    if window is not None:
        causal &= (i[:, None] - i[None, :]) < window
    same = (s[:, None, :] == s[:, :, None]) | (s[:, None, :] == 0)
    valid = (s != -1)[:, None, :]
    allow = (causal[None] & same & valid) | torch.eye(L, dtype=torch.bool, device=device)[None]
    return torch.zeros(len(segs), L, L, dtype=dtype, device=device).masked_fill(~allow, torch.finfo(dtype).min)[:, None]


class PointerHead(nn.Module):
    """q = W_q h[<decide>], k_i = W_k h[</opt>_i], logits_i = k_i·q / sqrt(dp).
    State-dict names (q.weight, q.bias, k.weight, k.bias) are what
    grande-core's safetensors loader expects."""

    def __init__(self, d: int, dp: int = 256):
        super().__init__()
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)

    def forward(self, h_decide, h_opts):  # [d], [K,d] -> [K]
        return (self.k(h_opts) @ self.q(h_decide)) * self.scale


class DecisionModel(nn.Module):
    def __init__(self, backbone, hidden_size: int, lora_r: int | None = 16, sliding_window: int | None = None):
        super().__init__()
        self.lm = backbone  # text model without lm_head
        if lora_r:
            from peft import LoraConfig, get_peft_model

            cfg = LoraConfig(
                r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            self.lm = get_peft_model(self.lm, cfg)
        self.head = PointerHead(hidden_size)
        self.sliding_window = sliding_window

    def hidden(self, encs):
        dev = next(self.parameters()).device
        L = max(len(e.ids) for e in encs)
        ids = torch.zeros((len(encs), L), dtype=torch.long, device=dev)
        pos = torch.zeros((len(encs), L), dtype=torch.long, device=dev)
        for b, e in enumerate(encs):
            ids[b, : len(e.ids)] = torch.tensor(e.ids, device=dev)
            pos[b, : len(e.pos)] = torch.tensor(e.pos, device=dev)
        segs = [e.seg for e in encs]
        full = branch_mask(segs, dev)
        if self.sliding_window:
            # transformers >= 4.53 accepts a per-layer-type mask mapping.
            mask = {"full_attention": full, "sliding_attention": branch_mask(segs, dev, self.sliding_window)}
        else:
            mask = full
        return self.lm(input_ids=ids, position_ids=pos, attention_mask=mask).last_hidden_state

    def forward(self, encs):
        hs = self.hidden(encs)
        out = []
        for b, e in enumerate(encs):
            out.append([self.head(hs[b, d], hs[b, torch.tensor(oi, device=hs.device)]) for d, oi in zip(e.decide_idx, e.opt_idx)])
        return out

    def loss(self, encs, labels, soft=None):
        """labels[b][k] = index of the gold option in rendered order (-1 to skip).
        soft[b][k], when given, is a target distribution in rendered order and
        the loss is KL(target || model) instead of cross-entropy."""
        logits = self(encs)
        losses = []
        for b in range(len(encs)):
            for k, z in enumerate(logits[b]):
                t = soft[b][k] if soft is not None else None
                if t is not None:
                    tt = torch.tensor(t, device=z.device, dtype=z.dtype)
                    losses.append(F.kl_div(F.log_softmax(z, -1), tt, reduction="sum"))
                    continue
                y = labels[b][k]
                if y >= 0:
                    losses.append(F.cross_entropy(z[None], torch.tensor([y], device=z.device)))
        return torch.stack(losses).mean()
