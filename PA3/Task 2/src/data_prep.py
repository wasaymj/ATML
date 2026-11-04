"""Data selection for Task 2 (all seeded with 6304; every selected id is saved).

Three components, built from COCO 2014 captions + instances and VQAv2 complementary pairs:
  * captions   : 6,000 train2014 images x 3 captions = 18,000 examples; 96 val2014 dev images x 1 caption.
  * grounding  : programmatically generated presence ("Does this image contain [object]?", yes/no) and counting
                 ("How many [objects] can you see?", two different counts in 1-4) pairs. Training pools are built
                 on train2014 images; dev 24 pairs / test 64 pairs on val2014.
  * vqa        : VQAv2 complementary pairs whose normalized consensus answers differ, with >= 8/10 agreement on
                 each image; 640/256/640 train, 12/8/12 dev, 36/24/36 test pairs (yes/no, number, other).

Split rules: training uses train2014 only; dev and final test use val2014 only. All five evaluation sets (caption
dev, grounding dev/test, VQA dev/test) are image-disjoint from one another, and every evaluation suite uses each
image once (so the mismatched-image control can exclude both images of the original pair). Training images may
recur across curriculum stages (the grounding pools reuse the caption and VQA training images).
"""
import os
import re
import gc
import json
import random
import zipfile
import logging
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set, Tuple

from src.utils import atomic_json_dump, ensure_dir, load_json, task_path
from src.vqa_eval import consensus, normalize_answer

logger = logging.getLogger(__name__)

# Plural forms used in counting questions; names already plural or uncountable take no article in presence questions.
PLURAL = {"person": "people", "bus": "buses", "bench": "benches", "couch": "couches", "sandwich": "sandwiches",
          "toothbrush": "toothbrushes", "wine glass": "wine glasses", "knife": "knives", "mouse": "mice",
          "sheep": "sheep", "skis": "skis", "scissors": "scissors", "broccoli": "broccoli", "glass": "glasses",
          "dining table": "dining tables", "hair drier": "hair driers", "skateboard": "skateboards",
          "sports ball": "sports balls", "fire hydrant": "fire hydrants", "stop sign": "stop signs"}
NO_ARTICLE = {"skis", "scissors", "broccoli"}
NOT_COUNTABLE = {"broccoli", "skis"}        # instances do not correspond to a natural count


def plural(name: str) -> str:
    if name in PLURAL:
        return PLURAL[name]
    if name.endswith(("s", "x", "ch", "sh")):
        return name + "es"
    return name + "s"


def with_article(name: str) -> str:
    if name in NO_ARTICLE:
        return name
    return ("an " if name[0] in "aeiou" else "a ") + name


def image_key(split: str, image_id: int) -> str:
    return f"{split}/{int(image_id)}"


def image_filename(key: str) -> str:
    split, iid = key.split("/")
    return f"{split}/COCO_{split}_{int(iid):012d}.jpg"


# --------------------------------------------------------------------------------------------
# Downloads
# --------------------------------------------------------------------------------------------
def _download(url: str, dest: str, retries: int = 4) -> None:
    tmp = dest + ".tmp"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            os.replace(tmp, dest)
            return
        except Exception as e:      # transient network errors: retry
            logger.warning("download %s failed (%s), attempt %d", url, e, attempt + 1)
    raise RuntimeError(f"could not download {url}")


RAW_FILES = {
    "vqa_q_train": "v2_OpenEnded_mscoco_train2014_questions.json",
    "vqa_a_train": "v2_mscoco_train2014_annotations.json",
    "vqa_p_train": "v2_mscoco_train2014_complementary_pairs.json",
    "vqa_q_val": "v2_OpenEnded_mscoco_val2014_questions.json",
    "vqa_a_val": "v2_mscoco_val2014_annotations.json",
    "vqa_p_val": "v2_mscoco_val2014_complementary_pairs.json",
    "cap_train": "annotations/captions_train2014.json",
    "cap_val": "annotations/captions_val2014.json",
    "inst_train": "annotations/instances_train2014.json",
    "inst_val": "annotations/instances_val2014.json",
}


def download_annotations(cfg: Dict[str, Any]) -> str:
    raw = ensure_dir(task_path(cfg["paths"]["raw_dir"]))
    d = cfg["downloads"]
    vqa_names = [RAW_FILES[k] for k in RAW_FILES if k.startswith("vqa")]
    if not all(os.path.exists(os.path.join(raw, n)) and os.path.getsize(os.path.join(raw, n)) > 0 for n in vqa_names):
        for z in d["vqa_files"]:
            zp = os.path.join(raw, z)
            logger.info("downloading %s", z)
            _download(d["vqa_base"] + z, zp)
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(raw)
            os.remove(zp)
    coco_names = [RAW_FILES[k] for k in RAW_FILES if k.startswith(("cap", "inst"))]
    if not all(os.path.exists(os.path.join(raw, n)) for n in coco_names):
        zp = os.path.join(raw, "annotations_trainval2014.zip")
        logger.info("downloading COCO 2014 annotations (~250 MB)")
        _download(d["coco_annotations"], zp)
        with zipfile.ZipFile(zp) as zf:
            for n in coco_names:
                zf.extract(n, raw)
        os.remove(zp)
    for n in RAW_FILES.values():
        if not os.path.exists(os.path.join(raw, n)) or os.path.getsize(os.path.join(raw, n)) == 0:
            raise RuntimeError(f"missing or empty raw file {n}")
    return raw


def locate_image(cfg: Dict[str, Any], key: str) -> Optional[str]:
    fn = image_filename(key)
    for root in [task_path(cfg["paths"]["image_dir"])] + list(cfg["paths"].get("local_image_dirs") or []):
        p = os.path.join(root, fn)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None


def download_images(cfg: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """Fetch only the selected images (about 9-10k) instead of the 19 GB COCO archives; verify each decodes."""
    from PIL import Image
    root = task_path(cfg["paths"]["image_dir"])
    todo = [k for k in keys if locate_image(cfg, k) is None]
    logger.info("images: %d needed, %d to download", len(keys), len(todo))

    def fetch(k: str) -> Optional[str]:
        dest = os.path.join(root, image_filename(k))
        ensure_dir(os.path.dirname(dest))
        try:
            _download(cfg["downloads"]["coco_images"] + image_filename(k), dest)
            with Image.open(dest) as im:
                im.verify()
            return None
        except Exception as e:
            return f"{k}: {e}"

    errors = []
    with ThreadPoolExecutor(cfg["downloads"]["download_workers"]) as ex:
        for i, err in enumerate(ex.map(fetch, todo)):
            if err:
                errors.append(err)
            if (i + 1) % 1000 == 0:
                logger.info("  downloaded %d/%d", i + 1, len(todo))
    if errors:
        raise RuntimeError(f"{len(errors)} images failed, e.g. {errors[:3]}")
    return {"n_images": len(keys), "n_downloaded": len(todo)}


# --------------------------------------------------------------------------------------------
# Raw loading (streamed: the train annotation files are 330-355 MB of JSON, so whole-file json.load needs several
# GB of RAM; we parse one array element at a time and keep only the fields we use)
# --------------------------------------------------------------------------------------------
def iter_json_array(path: str, key: str, chunk: int = 1 << 22):
    """Yield the elements of the top-level array stored under `key` without loading the whole file."""
    dec = json.JSONDecoder()
    pat = re.compile(r'"%s"\s*:\s*\[' % re.escape(key))
    with open(path, "r", encoding="utf-8") as f:
        buf = ""
        while True:
            m = pat.search(buf)
            if m:
                buf = buf[m.end():]
                break
            data = f.read(chunk)
            if not data:
                raise KeyError(f"{key} not found in {path}")
            buf = buf[-(len(key) + 16):] + data
        pos = 0
        while True:
            while pos < len(buf) and buf[pos] in " \t\r\n,":
                pos += 1
            if pos >= len(buf):
                more = f.read(chunk)
                if not more:
                    return
                buf, pos = more, 0
                continue
            if buf[pos] == "]":
                return
            try:
                obj, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:
                more = f.read(chunk)
                if not more:
                    raise
                buf, pos = buf[pos:] + more, 0
                continue
            yield obj
            pos = end
            if pos > chunk:
                buf, pos = buf[pos:], 0


def load_instances(path: str, keep: Optional[Set[int]] = None) -> Tuple[Dict[int, Dict], Dict[int, str]]:
    """image_id -> {"area": W*H, "cats": {cat_id: [(area, iscrowd), ...]}}, and cat_id -> name.
    keep: restrict to these image ids (training pools only need the caption/VQA training images)."""
    cats = {c["id"]: c["name"] for c in iter_json_array(path, "categories")}
    imgs = {im["id"]: {"area": float(im["width"] * im["height"]), "cats": defaultdict(list)}
            for im in iter_json_array(path, "images") if keep is None or im["id"] in keep}
    for a in iter_json_array(path, "annotations"):
        if a["image_id"] in imgs:
            imgs[a["image_id"]]["cats"][a["category_id"]].append((float(a["area"]), int(a["iscrowd"])))
    return imgs, cats


def load_captions(path: str) -> Dict[int, List[Tuple[int, str]]]:
    out: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    for a in iter_json_array(path, "annotations"):
        out[a["image_id"]].append((a["id"], a["caption"].strip()))
    for v in out.values():
        v.sort()
    return out


def load_vqa(raw: str, split: str, min_agree: int) -> Tuple[Dict[int, Dict], List[List[int]]]:
    """Compact records for questions that occur in a complementary pair. The consensus is computed while
    streaming; the ten raw answers are kept only for questions that pass the agreement rule (the others can never
    be selected), which keeps memory small."""
    pairs = load_json(os.path.join(raw, RAW_FILES[f"vqa_p_{split}"]))
    in_pair = {q for p in pairs for q in p}
    out: Dict[int, Dict] = {}
    for q in iter_json_array(os.path.join(raw, RAW_FILES[f"vqa_q_{split}"]), "questions"):
        if q["question_id"] in in_pair:
            out[q["question_id"]] = {"question": q["question"].strip(), "image_id": q["image_id"]}
    for a in iter_json_array(os.path.join(raw, RAW_FILES[f"vqa_a_{split}"]), "annotations"):
        rec = out.get(a["question_id"])
        if rec is None:
            continue
        answers = [x["answer"] for x in a["answers"]]
        rec["consensus"] = consensus(answers)
        rec["answers"] = answers if rec["consensus"][1] >= min_agree else None
        rec["answer_type"] = a["answer_type"]
        rec["question_type"] = a["question_type"]
    return out, pairs


# --------------------------------------------------------------------------------------------
# Eligibility
# --------------------------------------------------------------------------------------------
class TokenCounter:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.cache: Dict[str, int] = {}

    def __call__(self, text: str) -> int:
        if text not in self.cache:
            self.cache[text] = len(self.tok(text, add_special_tokens=False)["input_ids"])
        return self.cache[text]


def eligible_vqa_pairs(vqa: Dict[int, Dict], pairs: List[List[int]], split: str, cfg: Dict[str, Any],
                       ntok: TokenCounter, stats: Dict[str, int]) -> Dict[str, List[Dict]]:
    """Keep pairs with identical question text, differing normalized consensus answers and >= 8/10 agreement on
    each image; the training target is an original answer string from the consensus group."""
    caps, min_agree = cfg["data"]["caps"], cfg["data"]["vqa_min_agreement"]
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for q1, q2 in pairs:
        stats["pairs_total"] += 1
        a, b = vqa.get(q1), vqa.get(q2)
        if a is None or b is None or "consensus" not in a or "consensus" not in b or a["image_id"] == b["image_id"]:
            stats["drop_missing_or_same_image"] += 1
            continue
        if a["question"].lower() != b["question"].lower():
            stats["drop_question_text_differs"] += 1
            continue
        if a["answer_type"] != b["answer_type"]:
            stats["drop_answer_type_differs"] += 1
            continue
        ca, cb = a["consensus"], b["consensus"]
        if ca[1] < min_agree or cb[1] < min_agree or len(a["answers"]) != 10 or len(b["answers"]) != 10:
            stats["drop_agreement_below_8"] += 1
            continue
        if ca[0] == cb[0]:
            stats["drop_same_consensus"] += 1
            continue
        if ntok(a["question"]) > caps["question"] or max(ntok(ca[2]), ntok(cb[2])) > caps["answer"]:
            stats["drop_token_cap"] += 1
            continue
        items = []
        for qid, rec, c in ((q1, a, ca), (q2, b, cb)):
            items.append({"qid": str(qid), "image": image_key(split, rec["image_id"]), "answer": c[2],
                          "answer_norm": c[0], "agreement": c[1], "answers": rec["answers"]})
        by_type[a["answer_type"]].append({"pair_id": f"vqa_{q1}_{q2}", "suite": "vqa", "kind": a["answer_type"],
                                          "question": a["question"], "question_type": a["question_type"],
                                          "items": items})
        stats["eligible"] += 1
    return by_type


class GroundingIndex:
    """Per-category candidate images for presence and counting under the label-quality rules."""

    def __init__(self, imgs: Dict[int, Dict], cats: Dict[int, str], image_ids: List[int], cfg: Dict[str, Any]):
        d = cfg["data"]
        lo, hi = d["count_range"]
        self.cats = cats
        self.image_ids = sorted(image_ids)
        self.img_cats: Dict[int, Set[int]] = {}
        self.pos: Dict[int, List[int]] = defaultdict(list)       # presence positives
        self.count: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        for iid in self.image_ids:
            im = imgs[iid]
            self.img_cats[iid] = set(im["cats"].keys())
            for c, inst in im["cats"].items():
                if max(a for a, _ in inst) >= d["presence_min_area_frac"] * im["area"]:
                    self.pos[c].append(iid)
                if cats[c] in NOT_COUNTABLE or any(cr for _, cr in inst):
                    continue
                if min(a for a, _ in inst) < d["count_min_area_frac"] * im["area"]:
                    continue
                if lo <= len(inst) <= hi:
                    self.count[c][len(inst)].append(iid)
        self.share_negatives = d["presence_negative_shares_category"]
        self.by_cat_images: Dict[int, Set[int]] = defaultdict(set)
        for iid, cs in self.img_cats.items():
            for c in cs:
                self.by_cat_images[c].add(iid)

    def presence_pair(self, rng: random.Random, used: Set[int]) -> Optional[Tuple[int, int, int]]:
        cands = [c for c in sorted(self.pos) if any(i not in used for i in self.pos[c])]
        if not cands:
            return None
        c = rng.choice(cands)
        p = rng.choice([i for i in self.pos[c] if i not in used])
        negs: List[int] = []
        if self.share_negatives:
            others = self.img_cats[p] - {c}
            pool: Set[int] = set()
            for o in others:
                pool |= self.by_cat_images[o]
            negs = sorted(i for i in pool if c not in self.img_cats[i] and i not in used and i != p)
        if not negs:
            negs = [i for i in self.image_ids if c not in self.img_cats[i] and i not in used and i != p]
        if not negs:
            return None
        return c, p, rng.choice(negs)

    def counting_pair(self, rng: random.Random, used: Set[int]) -> Optional[Tuple[int, Tuple[int, int], Tuple[int, int]]]:
        def free(c, n):
            return [i for i in self.count[c][n] if i not in used]
        cands = [c for c in sorted(self.count) if sum(1 for n in self.count[c] if free(c, n)) >= 2]
        if not cands:
            return None
        c = rng.choice(cands)
        levels = [n for n in sorted(self.count[c]) if free(c, n)]
        n1, n2 = rng.sample(levels, 2)            # uniform over count levels, so no count dominates the targets
        i1 = rng.choice(free(c, n1))
        i2 = rng.choice([i for i in free(c, n2) if i != i1] or [None])
        if i2 is None:
            return None
        return c, (i1, n1), (i2, n2)


def make_grounding_pair(kind: str, split: str, cfg: Dict[str, Any], cats: Dict[int, str], sample) -> Dict:
    pr = cfg["prompts"]
    if kind == "presence":
        c, pos, neg = sample
        name = cats[c]
        q = pr["presence"].format(obj=with_article(name))
        items = [(pos, "yes"), (neg, "no")]
    else:
        c, (i1, n1), (i2, n2) = sample
        name = cats[c]
        q = pr["counting"].format(objs=plural(name))
        items = [(i1, str(n1)), (i2, str(n2))]
    pid = f"{kind}_{c}_{items[0][0]}_{items[1][0]}"
    return {"pair_id": pid, "suite": "grounding", "kind": kind, "category": name, "category_id": c, "question": q,
            "items": [{"qid": f"{pid}_{j}", "image": image_key(split, iid), "answer": ans,
                       "answer_norm": normalize_answer(ans)} for j, (iid, ans) in enumerate(items)]}


def sample_grounding(index: GroundingIndex, kind: str, n: int, split: str, cfg: Dict[str, Any], rng: random.Random,
                     used: Optional[Set[int]], disjoint: bool, max_tries: int = 200000) -> List[Dict]:
    """n distinct pairs. disjoint=True: no image is reused within the call or appears in `used` (evaluation);
    disjoint=False: pairs are only required to be distinct (training pools)."""
    out, seen = [], set()
    blocked = used if used is not None else set()
    tries = 0
    while len(out) < n:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(f"could only build {len(out)}/{n} {kind} pairs")
        s = index.presence_pair(rng, blocked if disjoint else set()) if kind == "presence" \
            else index.counting_pair(rng, blocked if disjoint else set())
        if s is None:
            continue
        p = make_grounding_pair(kind, split, cfg, index.cats, s)
        key = (p["category_id"], tuple(sorted(it["image"] for it in p["items"])))
        if key in seen:
            continue
        seen.add(key)
        if disjoint:
            ids = [int(it["image"].split("/")[1]) for it in p["items"]]
            blocked.update(ids)
        out.append(p)
    rng.shuffle(out)
    for p in out:                                  # random within-pair order (yes-image / larger count not always first)
        if rng.random() < 0.5:
            p["items"] = p["items"][::-1]
    return out


def pick_vqa_pairs(by_type: Dict[str, List[Dict]], counts: Dict[str, int], rng: random.Random,
                   used: Optional[Set[str]]) -> List[Dict]:
    """Seeded draw of the requested number of pairs per answer type; with `used`, images must be unused (and are
    then reserved) so evaluation suites stay image-disjoint."""
    out = []
    for t, n in counts.items():
        pool = list(by_type[t])
        rng.shuffle(pool)
        got = 0
        for p in pool:
            if got == n:
                break
            imgs = [it["image"] for it in p["items"]]
            if used is not None:
                if any(i in used for i in imgs):
                    continue
                used.update(imgs)
            out.append(p)
            got += 1
        if got < n:
            raise RuntimeError(f"only {got}/{n} eligible VQA pairs of type {t}")
    return out


def mismatch_mapping(pairs: List[Dict], rng: random.Random) -> Dict[str, str]:
    """qid -> image of a different pair in the same suite. A single random cycle over pairs guarantees that neither
    image of the question's own pair is used; fixed once and reused for every model and comparison."""
    n = len(pairs)
    perm = list(range(n))
    rng.shuffle(perm)
    target = {perm[i]: perm[(i + 1) % n] for i in range(n)}
    out = {}
    for k, p in enumerate(pairs):
        other = pairs[target[k]]
        for it in p["items"]:
            out[it["qid"]] = other["items"][rng.randrange(2)]["image"]
    return out


# --------------------------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------------------------
def prepare_data(cfg: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    d = cfg["data"]
    raw = download_annotations(cfg)
    build_dir = ensure_dir(task_path(cfg["paths"]["build_dir"]))
    ids_dir = ensure_dir(task_path(cfg["paths"]["ids_dir"]))
    seed = cfg["seed"]
    ntok = TokenCounter(tokenizer)
    stats: Dict[str, Any] = {"seed": seed}

    caps = d["caps"]

    def caption_ok(c: str) -> bool:
        return ntok(c) <= caps["caption"]

    # ---------------- evaluation sets (val2014), selected first, image-disjoint across all suites -----------
    # (val files are loaded, used and released before the larger train files are read)
    logger.info("selecting evaluation sets from val2014")
    rng = random.Random(seed)
    used_eval: Set[str] = set()
    st_va: Dict[str, int] = Counter()
    vqa_va, pairs_va = load_vqa(raw, "val", d["vqa_min_agreement"])
    vqa_va_by_type = eligible_vqa_pairs(vqa_va, pairs_va, "val2014", cfg, ntok, st_va)
    del vqa_va, pairs_va
    stats["vqa_val_filter"] = dict(st_va)
    stats["vqa_val_eligible_by_type"] = {t: len(v) for t, v in vqa_va_by_type.items()}
    vqa_test = pick_vqa_pairs(vqa_va_by_type, d["vqa_test_pairs"], rng, used_eval)
    vqa_dev = pick_vqa_pairs(vqa_va_by_type, d["vqa_dev_pairs"], rng, used_eval)
    del vqa_va_by_type
    gc.collect()

    inst_va, cats = load_instances(os.path.join(raw, RAW_FILES["inst_val"]))
    val_ids = sorted(inst_va.keys())
    g_val = GroundingIndex(inst_va, cats, val_ids, cfg)
    used_val_ids = {int(k.split("/")[1]) for k in used_eval}
    g_test, g_dev = [], []
    for kind in ("presence", "counting"):
        g_test += sample_grounding(g_val, kind, d["grounding_test_pairs"][kind], "val2014", cfg, rng, used_val_ids, True)
    for kind in ("presence", "counting"):
        g_dev += sample_grounding(g_val, kind, d["grounding_dev_pairs"][kind], "val2014", cfg, rng, used_val_ids, True)
    used_eval |= {image_key("val2014", i) for i in used_val_ids}
    del g_val, inst_va
    gc.collect()

    cap_va = load_captions(os.path.join(raw, RAW_FILES["cap_val"]))
    cap_dev_pool = [i for i in sorted(cap_va) if image_key("val2014", i) not in used_eval]
    rng.shuffle(cap_dev_pool)
    captions_dev = []
    for iid in cap_dev_pool:
        if len(captions_dev) == d["caption_dev_images"]:
            break
        ok = [x for x in cap_va[iid] if caption_ok(x[1])]
        if not ok:
            continue
        ann_id, text = rng.choice(ok)
        captions_dev.append({"ex_id": f"capdev_{ann_id}", "image": image_key("val2014", iid), "text": text,
                             "ann_id": ann_id})
        used_eval.add(image_key("val2014", iid))
    del cap_va
    gc.collect()

    # ---------------- training sets (train2014) --------------------------------------------------------------
    logger.info("selecting training sets from train2014")
    rng_tr = random.Random(seed + 1)
    cap_tr = load_captions(os.path.join(raw, RAW_FILES["cap_train"]))
    cap_tr_pool = sorted(cap_tr)
    rng_tr.shuffle(cap_tr_pool)
    captions_train = []
    n_img = 0
    for iid in cap_tr_pool:            # eligibility checked lazily in the seeded order
        if n_img == d["caption_train_images"]:
            break
        ok = [x for x in cap_tr[iid] if caption_ok(x[1])]
        if len(ok) < d["captions_per_train_image"]:
            continue
        for ann_id, text in rng_tr.sample(ok, d["captions_per_train_image"]):
            captions_train.append({"ex_id": f"cap_{ann_id}", "image": image_key("train2014", iid), "text": text,
                                   "ann_id": ann_id})
        n_img += 1
    del cap_tr
    gc.collect()

    st_tr: Dict[str, int] = Counter()
    vqa_tr, pairs_tr = load_vqa(raw, "train", d["vqa_min_agreement"])
    vqa_tr_by_type = eligible_vqa_pairs(vqa_tr, pairs_tr, "train2014", cfg, ntok, st_tr)
    del vqa_tr, pairs_tr
    stats["vqa_train_filter"] = dict(st_tr)
    stats["vqa_train_eligible_by_type"] = {t: len(v) for t, v in vqa_tr_by_type.items()}
    vqa_train = pick_vqa_pairs(vqa_tr_by_type, d["vqa_train_pairs"], rng_tr, None)
    del vqa_tr_by_type
    gc.collect()

    # Grounding pools reuse the caption and VQA training images (allowed: training images may recur across stages).
    g_img_ids = sorted({int(e["image"].split("/")[1]) for e in captions_train}
                       | {int(it["image"].split("/")[1]) for p in vqa_train for it in p["items"]})
    inst_tr, _ = load_instances(os.path.join(raw, RAW_FILES["inst_train"]), keep=set(g_img_ids))
    g_tr = GroundingIndex(inst_tr, cats, g_img_ids, cfg)
    pools = {kind: sample_grounding(g_tr, kind, d["grounding_pool_pairs"][kind], "train2014", cfg, rng_tr, None, False)
             for kind in ("presence", "counting")}

    # ---------------- fixed mismatched-image mappings (final test suites) ------------------------------------
    rng_mm = random.Random(seed + 2)
    mismatch = {"grounding_test": mismatch_mapping(g_test, rng_mm), "vqa_test": mismatch_mapping(vqa_test, rng_mm)}

    sets = {"captions_train": captions_train, "captions_dev": captions_dev,
            "grounding_pool_presence": pools["presence"], "grounding_pool_counting": pools["counting"],
            "grounding_dev": g_dev, "grounding_test": g_test,
            "vqa_train": vqa_train, "vqa_dev": vqa_dev, "vqa_test": vqa_test}
    checks = verify_splits(sets, mismatch, cfg)

    # ---------------- write: materialized sets (gitignored) + ids (git-tracked) -----------------------------
    for name, rows in sets.items():
        atomic_json_dump(rows, os.path.join(build_dir, f"{name}.json"))
    atomic_json_dump(mismatch, os.path.join(build_dir, "mismatch.json"))
    write_ids(sets, mismatch, ids_dir)

    stats.update(summarize_sets(sets))
    stats["grounding_label_rules"] = {k: d[k] for k in ("presence_min_area_frac", "count_min_area_frac",
                                                         "presence_negative_shares_category", "count_range")}
    stats["not_countable_categories"] = sorted(NOT_COUNTABLE)
    stats["grounding_train_candidate_images"] = len(g_img_ids)
    stats["checks"] = checks
    stats["max_tokens"] = {
        "question": max(ntok(p["question"]) for n in ("vqa_train", "vqa_dev", "vqa_test", "grounding_dev",
                                                       "grounding_test") for p in sets[n]),
        "caption": max(ntok(e["text"]) for e in captions_train + captions_dev),
        "answer": max(ntok(it["answer"]) for n in ("vqa_train", "vqa_dev", "vqa_test") for p in sets[n]
                      for it in p["items"])}
    return stats


def all_image_keys(build_dir: str) -> Dict[str, List[str]]:
    """Image keys per COCO split for everything that must be downloaded and cached."""
    keys: Dict[str, Set[str]] = {"train2014": set(), "val2014": set()}
    for fn in os.listdir(build_dir):
        if not fn.endswith(".json") or fn == "mismatch.json":
            continue
        for row in load_json(os.path.join(build_dir, fn)):
            for k in ([row["image"]] if "image" in row else [it["image"] for it in row["items"]]):
                keys[k.split("/")[0]].add(k)
    return {s: sorted(v, key=lambda k: int(k.split("/")[1])) for s, v in keys.items()}


def verify_splits(sets: Dict[str, List[Dict]], mismatch: Dict[str, Dict[str, str]], cfg: Dict[str, Any]) -> Dict:
    """Assert split isolation, sizes and pair properties; returns the evidence for the report."""
    d = cfg["data"]

    def images(name):
        rows = sets[name]
        return [r["image"] for r in rows] if rows and "image" in rows[0] else [it["image"] for r in rows for it in r["items"]]

    eval_sets = ["captions_dev", "grounding_dev", "grounding_test", "vqa_dev", "vqa_test"]
    train_sets = ["captions_train", "grounding_pool_presence", "grounding_pool_counting", "vqa_train"]
    out: Dict[str, Any] = {}
    for n in train_sets:
        assert all(k.startswith("train2014/") for k in images(n)), n
    for n in eval_sets:
        ims = images(n)
        assert all(k.startswith("val2014/") for k in ims), n
        assert len(ims) == len(set(ims)), f"{n}: an image is used twice within the suite"
    for i, a in enumerate(eval_sets):
        for b in eval_sets[i + 1:]:
            inter = set(images(a)) & set(images(b))
            assert not inter, f"{a} and {b} share images: {list(inter)[:3]}"
    train_imgs = set().union(*[set(images(n)) for n in train_sets])
    eval_imgs = set().union(*[set(images(n)) for n in eval_sets])
    assert not (train_imgs & eval_imgs)
    for n in ("grounding_dev", "grounding_test", "vqa_dev", "vqa_test", "vqa_train", "grounding_pool_presence",
              "grounding_pool_counting"):
        for p in sets[n]:
            a, b = p["items"]
            assert a["answer_norm"] != b["answer_norm"] and a["image"] != b["image"], p["pair_id"]
    for suite, mp in mismatch.items():
        own = {it["qid"]: {x["image"] for x in p["items"]} for p in sets[suite] for it in p["items"]}
        assert set(mp) == set(own)
        assert all(mp[q] not in own[q] for q in mp), f"{suite}: mismatched image from the question's own pair"
        assert set(mp.values()) <= set(images(suite))
    expect = {"captions_train": d["caption_train_images"] * d["captions_per_train_image"],
              "captions_dev": d["caption_dev_images"],
              "grounding_dev": sum(d["grounding_dev_pairs"].values()),
              "grounding_test": sum(d["grounding_test_pairs"].values()),
              "vqa_train": sum(d["vqa_train_pairs"].values()), "vqa_dev": sum(d["vqa_dev_pairs"].values()),
              "vqa_test": sum(d["vqa_test_pairs"].values())}
    for n, v in expect.items():
        assert len(sets[n]) == v, f"{n}: {len(sets[n])} != {v}"
    out["sizes_ok"] = expect
    out["train_images"] = len(train_imgs)
    out["eval_images"] = len(eval_imgs)
    out["eval_suites_image_disjoint"] = True
    out["train_eval_image_overlap"] = 0
    out["mismatch_excludes_own_pair"] = True
    return out


def summarize_sets(sets: Dict[str, List[Dict]]) -> Dict[str, Any]:
    s: Dict[str, Any] = {"sizes": {}, "answers": {}, "types": {}}
    for n, rows in sets.items():
        if n.startswith("captions"):
            s["sizes"][n] = {"examples": len(rows), "images": len({r["image"] for r in rows})}
            continue
        s["sizes"][n] = {"pairs": len(rows), "questions": 2 * len(rows),
                         "images": len({it["image"] for p in rows for it in p["items"]})}
        s["types"][n] = dict(Counter(p["kind"] for p in rows))
        ans = Counter(it["answer_norm"] for p in rows for it in p["items"])
        s["answers"][n] = {"n_distinct": len(ans), "top10": ans.most_common(10)}
        if n.startswith("grounding"):
            s.setdefault("grounding_categories", {})[n] = len({p["category"] for p in rows})
    return s


def write_ids(sets: Dict[str, List[Dict]], mismatch: Dict[str, Dict[str, str]], ids_dir: str) -> None:
    """Git-tracked record of every selected image / question / pair id (dataset text is not committed)."""
    for n, rows in sets.items():
        if n.startswith("captions"):
            out = [{"ex_id": r["ex_id"], "image": r["image"], "coco_caption_id": r["ann_id"]} for r in rows]
        elif n.startswith("vqa"):
            out = [{"pair_id": p["pair_id"], "kind": p["kind"], "question_ids": [it["qid"] for it in p["items"]],
                    "images": [it["image"] for it in p["items"]], "targets": [it["answer"] for it in p["items"]]}
                   for p in rows]
        else:      # generated questions: the full pair is small and is our own data
            out = rows
        atomic_json_dump(out, os.path.join(ids_dir, f"{n}.json"))
    atomic_json_dump(mismatch, os.path.join(ids_dir, "mismatch_mapping.json"))


def load_sets(cfg: Dict[str, Any]) -> Dict[str, Any]:
    b = task_path(cfg["paths"]["build_dir"])
    names = ["captions_train", "captions_dev", "grounding_pool_presence", "grounding_pool_counting", "grounding_dev",
             "grounding_test", "vqa_train", "vqa_dev", "vqa_test"]
    out = {n: load_json(os.path.join(b, f"{n}.json")) for n in names}
    out["mismatch"] = load_json(os.path.join(b, "mismatch.json"))
    return out


def label_audit_sheet(cfg: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    """Blind label check of the final-test grounding questions: contact sheets show image + question only (no
    annotation label, no prediction); a CSV is filled by hand and compared with the annotation-derived labels
    by the report compiler."""
    import csv
    from PIL import Image, ImageDraw
    sets = load_sets(cfg)
    rows = [(p, it) for p in sets["grounding_test"] for it in p["items"]]
    order = list(range(len(rows)))
    random.Random(cfg["seed"] + 3).shuffle(order)       # shuffled so pair partners are not adjacent
    ensure_dir(out_dir)
    per_page, cols, size = 16, 4, 300
    sheets = []
    for page in range(0, len(order), per_page):
        idx = order[page:page + per_page]
        canvas = Image.new("RGB", (cols * size, ((len(idx) + cols - 1) // cols) * (size + 60)), "white")
        draw = ImageDraw.Draw(canvas)
        for j, k in enumerate(idx):
            p, it = rows[k]
            im = Image.open(locate_image(cfg, it["image"])).convert("RGB")
            im.thumbnail((size - 10, size - 10))
            x, y = (j % cols) * size, (j // cols) * (size + 60)
            canvas.paste(im, (x + 5, y + 5))
            draw.text((x + 5, y + size - 2), f"#{k:03d}", fill="black")
            draw.text((x + 5, y + size + 14), p["question"][:48], fill="black")
        path = os.path.join(out_dir, f"audit_page_{page // per_page + 1}.png")
        canvas.save(path)
        sheets.append(path)
    with open(os.path.join(out_dir, "label_audit_blind.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "question", "your_answer (yes/no or a count; '?' if ambiguous)"])
        for k in sorted(order):
            w.writerow([f"{k:03d}", rows[k][0]["question"], ""])
    key = [{"audit_id": f"{k:03d}", "qid": rows[k][1]["qid"], "pair_id": rows[k][0]["pair_id"],
            "kind": rows[k][0]["kind"], "label": rows[k][1]["answer"]} for k in range(len(rows))]
    atomic_json_dump(key, os.path.join(out_dir, "label_audit_key.json"))
    return {"pages": sheets, "n_questions": len(rows)}
