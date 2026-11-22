"""CPU unit tests for Task 2 (python tests/test_units.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import attention_ops as A  # noqa: E402
from src import kv_quant as KQ  # noqa: E402
from src.mqar_model import DeltaAttention, LinearAttention, head_masks  # noqa: E402

torch.manual_seed(0)


def _qkv(B=2, H=3, L=100, d=16, dt=torch.float64):
    return [torch.randn(B, H, L, d, dtype=dt) for _ in range(4)]


def test_tiled_matches_dense_fwd_bwd():
    for L in (1, 37, 100, 128):
        for bq, bk in ((32, 32), (16, 64), (64, 16), (128, 128)):
            q, k, v, do = _qkv(L=L)
            ref = []
            for fn in (lambda a, b, c: A.dense_attention(a, b, c),
                       lambda a, b, c: A.tiled_attention(a, b, c, bq, bk),
                       lambda a, b, c: A.tiled_attention_recompute(a, b, c, bq, bk)):
                qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
                o = fn(qq, kk, vv)
                o.backward(do)
                ref.append((o.detach(), qq.grad, kk.grad, vv.grad))
            for other in ref[1:]:
                for a, b in zip(ref[0], other):
                    assert (a - b).abs().max() < 1e-10, (L, bq, bk, float((a - b).abs().max()))


def test_dense_matches_sdpa_and_masked_row():
    q, k, v, _ = _qkv(dt=torch.float32)
    assert (A.dense_attention(q, k, v) - A.sdpa(q, k, v)).abs().max() < 1e-5
    km = torch.ones(1, 1, 100, 100, dtype=torch.bool)
    km[..., 5, :] = False
    o = A.dense_attention(q, k, v, key_mask=km)
    assert torch.isfinite(o).all() and o[..., 5, :].abs().max() == 0


def test_causality_exact():
    q, k, v, _ = _qkv(dt=torch.float32)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 51:] += 3
    v2[:, :, 51:] -= 3
    for fn in (A.dense_attention, lambda a, b, c: A.tiled_attention(a, b, c, 16, 16)):
        assert torch.equal(fn(q, k, v)[:, :, :51], fn(q, k2, v2)[:, :, :51])


def test_linear_parallel_equals_recurrent():
    m = LinearAttention(32, 4, 1e-6).double()
    x = torch.randn(2, 50, 32, dtype=torch.float64)
    m.mode = "parallel"
    a = m(x)
    m.mode = "recurrent"
    b = m(x)
    assert (a - b).abs().max() < 1e-10


def test_delta_rule_naive():
    m = DeltaAttention(32, 4, 9).double()
    x = torch.randn(1, 12, 32, dtype=torch.float64)
    out = m(x)
    q = m._phi(m.q(x).view(1, 12, 4, 8).transpose(1, 2))
    k = m._phi(m.k(x).view(1, 12, 4, 8).transpose(1, 2))
    v = m.v(x).view(1, 12, 4, 8).transpose(1, 2)
    beta = torch.sigmoid(m.beta(x)).transpose(1, 2)
    outs = torch.zeros(1, 4, 12, 8, dtype=torch.float64)
    for h in range(4):
        S = torch.zeros(9, 8, dtype=torch.float64)
        for t in range(12):
            kt, vt = k[0, h, t], v[0, h, t]
            S = S + beta[0, h, t] * torch.outer(kt, vt - S.T @ kt)
            outs[0, h, t] = S.T @ q[0, h, t]
    ref = m.o(outs.transpose(1, 2).reshape(1, 12, 32))
    assert (out - ref).abs().max() < 1e-10


def test_masks():
    T = 256
    w = head_masks("window", T, 4, 64)
    assert int(w[0].sum(1).max()) == 64 and bool(w[0][200, 137]) and not bool(w[0][200, 136])
    d = head_masks("dilated", T, 4, 64, 4, 64)
    assert torch.equal(d[0], w[0]) and int(d[2].sum(1).max()) == 64
    assert bool(d[2][255, 3]) and not bool(d[2][255, 4]) and not bool(d[2][255, 2])     # 252 back, offsets x4


def test_kv_quant_bytes_and_roundtrip():
    for L in (64, 130, 2048, 2101):
        k, v = torch.randn(2, 2, L, 128).half(), torch.randn(2, 2, L, 128).half()
        for layout in ("channelK_tokenV", "tokenwise"):
            for b in (2, 4):
                enc = KQ.encode_layer(k, v, layout, b)
                assert KQ.encoded_bytes(enc) == 2 * KQ.bytes_quant(L, b, 1, 2, 128)     # batch 2
                dk, dv = KQ.decode_layer(enc, b)
                kh, vh, lq = KQ.quantize_kv(k, v, layout, b)
                assert torch.equal(dk, kh) and torch.equal(dv, vh)
                assert torch.equal(kh[:, :, lq:], k[:, :, lq:])                         # FP16 residual


def test_tail_quant_cache():
    L = 100
    k, v = torch.randn(1, 2, L, 128), torch.randn(1, 2, L, 128)
    kh, vh, lq = KQ.quantize_kv(k, v, "channelK_tokenV", 4)
    c = KQ.build_cache([(kh, vh)], KQ.TailQuantCache).setup("channelK_tokenV", 4, [lq])
    new = torch.randn(1, 2, 30, 128)
    c.update(new, new.clone(), 0)
    assert c.qend[0] == 128                                            # residual 36 + 30 = 66 -> one block quantized
    full_k = torch.cat([k, new], 2)
    exp = KQ.quantize_kv(full_k, torch.cat([v, new], 2), "channelK_tokenV", 4)[0]
    assert torch.allclose(c.layers[0].keys[:, :, :128], exp[:, :, :128])
    assert torch.equal(c.layers[0].keys[:, :, 128:], full_k[:, :, 128:])


def test_budget_sizes():
    from src.part_c import evict_sizes, hybrid_size
    cfg = {"part_c": {"sink": 8, "recent": 64, "index_bytes": 2, "bits_hybrid_low": 2}}
    for L in (1990, 2048, 4100):
        budget = KQ.bytes_quant(L, 4, 28, 2, 128)
        n_sr, n_qa = evict_sizes(cfg, L, budget, 28, 2, 128)
        assert n_sr * 28 * 2 * 512 <= budget < (n_sr + 1) * 28 * 2 * 512
        assert 56 * (n_qa * 512 + (n_qa - 72) * 2) <= budget
        M, Gc, r, nb = hybrid_size(cfg, L, budget, 28, 2, 128)
        assert nb <= budget and 72 + M + r + 64 * Gc == L


def test_diag_relocation_and_supervision():
    """Diagnostic transforms: the relocated K = 4 block keeps every key unique (checked inside shifted_k4) and the
    answers/distances consistent; supervision masks only remove labels."""
    import numpy as np
    from src import diag_b as D, mqar_data as MD
    from src.utils import load_config
    cfg = load_config()
    test = D.subsample(MD.load_frozen(cfg, "test"), 40)
    for s in D.SHIFTS:
        arr, qt = D.shifted_k4(test, s)
        tok = arr["tokens"].astype(int)
        for i, row in enumerate(arr["qpos"]):
            for c, p in enumerate(row):
                if p < 0:
                    continue
                key = tok[i, p]
                kpos = np.nonzero(tok[i, :p] == key)[0]
                assert len(kpos) == 1 and kpos[0] >= 2 * s and tok[i, kpos[0] + 1] == arr["answers"][i, c]
        assert (qt["slot"] >= s).all() and (qt["slot"] <= s + 3).all() and (qt["distance"] >= 1).all()
    pool = {k: test[k] for k in ("tokens", "qpos", "answers", "distances")}
    reach = D.supervision(pool, "reach")
    assert ((reach["qpos"] >= 0) <= (pool["qpos"] >= 0)).all()
    assert (pool["distances"][reach["qpos"] >= 0] <= D.REACH).all()
    thin = D.supervision(pool, "thin", keep_frac=0.25)
    assert ((thin["qpos"] >= 0) <= (pool["qpos"] >= 0)).all()


def test_diag_tokenwise_key_arms():
    """Outlier arm: kept channels are exact; the remaining channels are quantized on a grid that ignores them."""
    from src import diag_c as DC
    torch.manual_seed(0)
    k = torch.randn(1, 2, 130, 128, dtype=torch.float32)
    k[..., 5] += 40.0                                          # one outlier channel per head
    lq = 128
    out = DC._outlier(k, lq, 2)
    idx = DC.outlier_channels(k, lq, 2)
    assert (idx == 5).any(-1).all()
    assert torch.equal(out[:, :, :, 5], k[:, :, :, 5]) and torch.equal(out[:, :, lq:], k[:, :, lq:])
    e_out = (out[..., 64:] - k[..., 64:]).abs().mean()
    e_tok = (DC._token_k(k, lq)[..., :64] - k[..., :64]).abs().mean()
    assert (out[..., :64] - k[..., :64]).abs().mean() < 0.5 * e_tok and e_out < 0.5 * e_tok * 10
    c = DC._center(k, lq)
    assert torch.equal(c[:, :, lq:], k[:, :, lq:])
    for arm in DC.ARMS:
        kv, extra = DC.arm_kv(arm, [(k.half(), k.half())])
        assert kv[0][0].shape == k.shape and extra >= 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
