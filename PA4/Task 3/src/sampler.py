"""Exact speculative sampling (Leviathan et al., 2023, Algorithm 1) and plain ancestral sampling, batch size 1.

One implementation serves the toy Markov models (chi-square proof of exactness) and the real Qwen models, through a
small language-model interface:

    lm.reset()                                  empty cache
    lm.feed(tokens) -> probs (len(tokens), V)   next-token distributions after each fed token; the tokens are appended
                                                to the model's cache
    lm.crop(n)                                  keep only the first n cached tokens (rollback)
    lm.length                                   number of cached tokens
    lm.n_forward                                forward passes since reset

Cache convention ("pending last token"): each model's cache holds committed tokens only, and never the most recent
committed token; whatever is committed but not cached is fed at the start of the next call. For the target this is
always exactly one token, so a verification pass is ONE forward on [last committed, x_1..x_g] whose g+1 outputs are
p_1..p_{g+1}. After a round both caches are cropped to the committed prefix (rollback); the draft may lag by up to
two tokens (x_g and the bonus/residual token after a fully accepted round), which it feeds in its next catch-up pass.

Round with g = min(gamma, remaining - 1) proposals (so a round never overshoots the token cap):
  draft:   catch up on committed tokens, then sample x_1..x_g from q_1..q_g (g draft forward passes)
  target:  one forward -> p_1..p_{g+1}
  accept:  for i = 1..g, r ~ U(0,1); accept x_i iff r < p_i(x_i) / q_i(x_i); stop at the first rejection or at an
           accepted EOS
  reject:  sample t ~ norm(max(0, p_i - q_i))  (normalised positive residual)
  bonus:   all g accepted and no EOS -> t ~ p_{g+1}   (g = 0 is therefore an ordinary target step)
  commit:  emitted = accepted proposals + [t]; truncate after the first EOS; crop both caches.
residual="p" is a deliberately incorrect variant (resample from p instead of the residual): the negative control.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


# --------------------------------------------------------------------------------------- language models
class HFLM:
    """A Hugging Face causal LM with a DynamicCache (batch 1, no padding), returning FP32 next-token probabilities."""

    def __init__(self, model, temperature: float = 1.0):
        self.model = model
        self.T = temperature
        self.dev = next(model.parameters()).device
        self.cache = None
        self.n_forward = 0

    def reset(self):
        from transformers import DynamicCache
        self.cache = DynamicCache()
        self.n_forward = 0

    @property
    def length(self) -> int:
        return 0 if self.cache is None else self.cache.get_seq_length()

    @torch.no_grad()
    def feed(self, tokens):
        ids = torch.as_tensor([list(tokens)], device=self.dev)
        out = self.model(input_ids=ids, past_key_values=self.cache, use_cache=True)
        self.n_forward += 1
        return torch.softmax(out.logits[0].float() / self.T, dim=-1)

    def crop(self, n: int):
        if self.length > n:
            self.cache.crop(n)


class ToyMarkovLM:
    """Second-order Markov model over a small alphabet: probs(next | ..., a, b) = P[a, b], where b is the token just
    fed and a the cached token before it (the start symbol S if there is none). Because a is read from the cached
    history, a wrong rollback changes the distribution and is caught by the chi-square test."""

    def __init__(self, P: torch.Tensor, start: int):
        assert P.dim() == 3
        self.P = P.float()
        self.S = start
        self.hist = []
        self.n_forward = 0
        self.dev = "cpu"

    def reset(self):
        self.hist = []
        self.n_forward = 0

    @property
    def length(self) -> int:
        return len(self.hist)

    def feed(self, tokens):
        rows = []
        for t in (int(t) for t in tokens):
            a = self.hist[-1] if self.hist else self.S
            self.hist.append(t)
            rows.append(self.P[a, t])
        self.n_forward += 1
        return torch.stack(rows)

    def crop(self, n: int):
        del self.hist[n:]


# ------------------------------------------------------------------------------------------- sampling ops
def _sample(probs: torch.Tensor, gen) -> int:
    return int(torch.multinomial(probs, 1, generator=gen))


def _uniform(gen, device) -> float:
    return float(torch.rand((), generator=gen, device=device))


class _Clock:
    """Accumulating phase timer; synchronises CUDA on both sides of a phase when timing is on."""

    def __init__(self, on: bool, cuda: bool):
        self.on, self.cuda = on, cuda
        self.t = {}

    def start(self):
        if self.on and self.cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    def stop(self, key, t0):
        if self.on:
            if self.cuda:
                torch.cuda.synchronize()
            self.t[key] = self.t.get(key, 0.0) + time.perf_counter() - t0


@dataclass
class GenResult:
    tokens: list
    termination: str                      # "eos" | "cap"
    rounds: int = 0                       # speculative: target verification passes; plain: decode steps
    proposed: int = 0                     # draft proposals made (sum of g)
    examined: int = 0                     # proposals that reached the accept/reject test
    accepted: int = 0
    bonus: int = 0                        # rounds with g >= 1 whose proposals were all accepted -> bonus token
    plain_steps: int = 0                  # rounds with g = 0 (one token left before the cap): an ordinary target step
    residual_fallbacks: int = 0           # rejection with zero residual mass (p == q numerically); sampled from p
    draft_forwards: int = 0
    target_forwards: int = 0              # incl. the prefill pass
    accepted_per_round: list = field(default_factory=list)
    proposals: list = field(default_factory=list)   # per examined proposal: [position, index, accepted, p(x), q(x), TV]
    times: dict = field(default_factory=dict)        # seconds per phase (only with timing=True)
    wall: float = 0.0                                 # seconds, whole call


# --------------------------------------------------------------------------------------------- samplers
def sample_plain(lm, prompt, max_new, eos, gen, ignore_eos=False, timing=False, cuda=False) -> GenResult:
    """Ancestral sampling from one model: target-only baseline, teacher/student rollouts, prefix banks."""
    assert max_new >= 1 and len(prompt) >= 1
    clock = _Clock(timing, cuda)
    t_wall = time.perf_counter()
    lm.reset()
    t0 = clock.start()
    if len(prompt) > 1:
        lm.feed(prompt[:-1])
    clock.stop("prefill", t0)
    out, last, term = [], prompt[-1], "cap"
    while len(out) < max_new:
        t0 = clock.start()
        p = lm.feed([last])[-1]
        clock.stop("step_forward", t0)
        t0 = clock.start()
        last = _sample(p, gen)
        clock.stop("sample", t0)
        out.append(last)
        if last in eos and not ignore_eos:
            term = "eos"
            break
    if cuda:
        torch.cuda.synchronize()
    return GenResult(tokens=out, termination=term, rounds=len(out), target_forwards=lm.n_forward, times=clock.t,
                     wall=time.perf_counter() - t_wall)


def sample_speculative(target, draft, prompt, gamma, max_new, eos, gen, ignore_eos=False, residual="correct",
                       record=False, timing=False, cuda=False) -> GenResult:
    """Exact speculative sampling; `record` keeps one row per examined proposal (incl. TV(p_i, q_i))."""
    assert gamma >= 1 and max_new >= 1 and len(prompt) >= 1 and residual in ("correct", "p")
    clock = _Clock(timing, cuda)
    t_wall = time.perf_counter()
    target.reset()
    draft.reset()
    dev = target.dev
    t0 = clock.start()
    if len(prompt) > 1:
        target.feed(prompt[:-1])
        draft.feed(prompt[:-1])
    clock.stop("prefill", t0)
    committed = list(prompt)
    res = GenResult(tokens=[], termination="cap")
    while len(res.tokens) < max_new:
        g = min(gamma, max_new - len(res.tokens) - 1)
        xs, qs = [], []
        # ---- draft proposals
        t0 = clock.start()
        if g > 0:
            q = draft.feed(committed[draft.length:])[-1]               # catch-up pass also yields q_1
            for i in range(g):
                x = _sample(q, gen)
                xs.append(x)
                qs.append(q)
                if i < g - 1:
                    q = draft.feed([x])[-1]
        clock.stop("draft", t0)
        # ---- one target verification pass
        t0 = clock.start()
        pending = committed[target.length:]
        assert len(pending) == 1, "target cache out of sync"
        P = target.feed(pending + xs)                                  # rows: p_1 .. p_{g+1}
        clock.stop("verify", t0)
        # ---- accept / reject / residual / bonus
        t0 = clock.start()
        new, acc, rejected, eos_hit = [], 0, False, False
        for i in range(g):
            x, p_i, q_i = xs[i], P[i], qs[i]
            res.examined += 1
            ok = _uniform(gen, dev) < float(p_i[x] / q_i[x])
            if record:
                res.proposals.append([len(res.tokens) + i + 1, i + 1, int(ok), float(p_i[x]), float(q_i[x]),
                                      float(0.5 * (p_i - q_i).abs().sum())])
            if ok:
                new.append(x)
                acc += 1
                if x in eos and not ignore_eos:
                    eos_hit = True
                    break
            else:
                if residual == "correct":
                    r = torch.clamp(p_i - q_i, min=0.0)
                    mass = float(r.sum())
                    if mass <= 0.0:
                        res.residual_fallbacks += 1
                        r, mass = p_i, 1.0
                    new.append(_sample(r / mass, gen))
                else:
                    new.append(_sample(p_i, gen))
                rejected = True
                break
        if not rejected and not eos_hit:
            new.append(_sample(P[g], gen))
            if g > 0:
                res.bonus += 1
            else:
                res.plain_steps += 1
        clock.stop("accept", t0)
        res.rounds += 1
        res.proposed += g
        res.accepted += acc
        res.accepted_per_round.append(acc)
        if not ignore_eos:
            for j, tok in enumerate(new):
                if tok in eos:
                    new = new[:j + 1]
                    res.termination = "eos"
                    break
        assert len(res.tokens) + len(new) <= max_new
        committed.extend(new)
        res.tokens.extend(new)
        # ---- rollback / advance: keep exactly the committed tokens that are cached
        t0 = clock.start()
        target.crop(len(committed) - 1)
        draft.crop(min(draft.length, len(committed) - 1))
        clock.stop("rollback", t0)
        if res.termination == "eos":
            break
    if cuda:
        torch.cuda.synchronize()
    res.draft_forwards, res.target_forwards = draft.n_forward, target.n_forward
    res.times, res.wall = clock.t, time.perf_counter() - t_wall
    return res
