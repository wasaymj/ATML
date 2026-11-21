"""Fast CPU unit tests of the Task 1 building blocks (python -m pytest tests/test_units.py -q, or run directly)."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pruning as P  # noqa: E402
from src import quant as Q  # noqa: E402
from src import sensitivity as S  # noqa: E402


def test_quantizer_bounds_and_packing():
    torch.manual_seed(0)
    W = (torch.randn(64, 384) * 0.05).half()
    for b in (2, 3, 4, 8):
        deq, codes, zeros, sc, st = Q.quantize(W, b, 128, 1e-5, return_codes=True)
        assert codes.max() <= 2 ** b - 1 and zeros.max() <= 2 ** b - 1
        step = sc.float().repeat_interleave(128, 1)
        err = (deq.float() - W.float()).abs()
        # |w - w_hat| <= s/2 inside the range; up to one extra step at the group max from FP16 scale rounding
        assert float((err / step).max()) <= 1.0 + 1e-3
        for x, n in ((codes, 384), (zeros, 3)):
            pk = Q.pack_bits(x, b)
            assert pk.shape[1] == -(-n * b // 8)
            assert torch.equal(Q.unpack_bits(pk, b, n), x)
        assert torch.equal(Q.dequant_from_codes(codes, zeros, sc, 128, W.dtype), deq)
        assert Q.packed_bytes(64, 384, b) == Q.pack_bits(codes, b).numel() + Q.pack_bits(zeros, b).numel() + sc.numel() * 2


def test_quantizer_matches_manual_formula():
    W = torch.tensor([[-0.3, -0.1, 0.0, 0.25] * 32], dtype=torch.float32)
    deq = Q.quantize(W, 4, 128)
    s = torch.tensor((0.25 + 0.3) / 15).half().float()
    z = torch.clamp(torch.round(torch.tensor(0.3) / s), 0, 15)
    ref = s * (torch.clamp(torch.round(W / s) + z, 0, 15) - z)
    assert torch.allclose(deq, ref)


def test_row_counts_and_masks():
    torch.manual_seed(1)
    sc = torch.rand(10, 16)
    c = P.row_counts(37, 10, 16)
    assert int(c.sum()) == 37 and c.max() - c.min() <= 1
    m = P.per_row_mask(sc, c)
    assert torch.equal(m.sum(1), c)
    # pruned weights are the lowest scores of their row
    for i in range(10):
        assert sc[i][m[i]].max() <= sc[i][~m[i]].min()
    nm = P.nm_mask(sc, 2, 4)
    assert torch.all(nm.reshape(10, 4, 4).sum(-1) == 2)


def test_owl_allocation_mean_and_bounds():
    rng = np.random.default_rng(0)
    for S_ in (0.5, 0.75):
        for lam in (0.04, 0.12, 0.3):
            v = rng.random(28)
            a = np.array(S.owl_allocation(v, S_, lam, 0.10, 0.85))
            assert abs(a.mean() - S_) < 1e-9 and a.min() >= 0.10 - 1e-12 and a.max() <= 0.85 + 1e-12
            # monotone: more sensitive -> not more sparse
            o = np.argsort(v)
            assert np.all(np.diff(a[o]) <= 1e-9)


def test_knapsack_matches_brute_force():
    rng = np.random.default_rng(2)
    for _ in range(20):
        U, K = 5, 4
        cost = np.sort(rng.integers(1, 20, size=(U, K)) * 512, axis=1)
        dmg = -np.sort(-rng.random((U, K)), axis=1)
        budget = int(cost[:, 1].sum())
        pick, info = S.knapsack(dmg, cost, budget)
        best = min((sum(dmg[u, k] for u, k in enumerate(ch)), ch) for ch in itertools.product(range(K), repeat=U)
                   if sum(cost[u, k] for u, k in enumerate(ch)) <= budget)
        assert abs(info["objective"] - best[0]) < 1e-12 and info["total_cost"] <= budget and info["exact"]


def test_awq_fold_preserves_function():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=300, hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, tie_word_embeddings=True)
    m = Qwen2ForCausalLM(cfg).eval()
    for blk in m.model.layers:
        torch.nn.init.normal_(blk.self_attn.v_proj.bias, 0, 0.1)
    ids = torch.randint(0, 300, (2, 32))
    with torch.no_grad():
        ref = m(input_ids=ids).logits
        table = {l: {g: {4: {"scales": torch.rand(128 if g == "o" else (512 if g == "down" else 256)) + 0.5}}
                     for g in Q.GROUP_NAMES} for l in range(2)}
        Q.apply_awq(m, table, {(l, g): 4 for l in range(2) for g in Q.GROUP_NAMES}, sink=None)
        out = m(input_ids=ids).logits
    assert (out - ref).abs().max() < 1e-4, float((out - ref).abs().max())


def test_exact_zero_counts():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(vocab_size=100, hidden_size=256, intermediate_size=512, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2)
    m = Qwen2ForCausalLM(cfg)
    N = sum(p.weight.numel() for b in m.model.layers for p in
            [b.self_attn.q_proj, b.self_attn.k_proj, b.self_attn.v_proj, b.self_attn.o_proj, b.mlp.gate_proj,
             b.mlp.up_proj, b.mlp.down_proj])
    for S_ in (0.5, 0.75):
        a = S.owl_allocation([0.1, 0.7, 0.3], S_, 0.1, 0.1, 0.85)
        c = P.exact_zero_counts(m, a, S_)
        assert sum(c.values()) == round(S_ * N)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
