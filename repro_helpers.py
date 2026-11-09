"""Validation, post-processing, and scoring for OmniDiff evaluation and improvements."""
import hashlib
import json
from pathlib import Path
import re

COUNTS = {"Real": 803, "Render": 767}
PAPER = {
    "Real": {"BLEU-4": 14.3, "METEOR": 18.9, "ROUGE-L": 32.9, "CIDEr": 31.3},
    "Render": {"BLEU-4": 15.7, "METEOR": 19.9, "ROUGE-L": 35.3, "CIDEr": 28.3},
}
BASELINE_REPRODUCED = {
    "Real": {"BLEU-4": 14.67, "METEOR": 19.54, "ROUGE-L": 32.80, "CIDEr": 36.28},
    "Render": {"BLEU-4": 17.16, "METEOR": 20.17, "ROUGE-L": 35.59, "CIDEr": 30.89},
}

_ORDINAL_HALLUCINATION_RE = re.compile(
    r"\b(?:seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|"
    r"fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|"
    r"twenty-\w+|thirtieth|thirty-\w+|fortieth|forty-\w+|fiftieth|fifty-\w+)\s+row\b",
    re.IGNORECASE,
)


def postprocess_caption(text: str) -> str:
    """Clean degenerate repetition loops, counting hallucinations, and punctuation spacing bugs.

    Addresses three systematic defects observed in greedy M3Diff decoding:
    1. Missing whitespace after period (e.g. 'background.The' -> 'background. The'),
       which corrupts PTBTokenizer word boundaries.
    2. Degenerate sentence-level loops where greedy decoding repeats sentences
       until hitting max_new_tokens=1024 (~980 words).
    3. Ordinal row-counting hallucinations (e.g. 'in the fifty-third row')
       and truncated trailing fragments at the token limit.
    """
    if not isinstance(text, str) or not text.strip():
        return text
    # 1. Fix missing space after sentence-ending period before a capital letter
    cleaned = re.sub(r"\.([A-Z])", r". \1", text.strip())
    cleaned = re.sub(r"\s+", " ", cleaned)

    # 2. Split into sentences and remove duplicate/looping or hallucinated sentences
    raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    kept = []
    seen_normalized = set()
    for idx, sent in enumerate(raw_sentences):
        # Drop truncated trailing sentence fragment when caption hit max_new_tokens
        if idx == len(raw_sentences) - 1 and len(raw_sentences) > 5 and not sent.endswith((".", "!", "?")):
            continue
        # Drop ordinal counting hallucinations (e.g. seventh row ... fifty-third row)
        if len(kept) >= 4 and _ORDINAL_HALLUCINATION_RE.search(sent):
            continue
        norm = re.sub(r"\s+", " ", sent.lower().rstrip(".!?")).strip()
        if norm in seen_normalized:
            continue
        seen_normalized.add(norm)
        kept.append(sent)

    result = " ".join(kept).strip()
    return result if result else text.strip()


def postprocess_predictions(rows):
    """Return a new list of prediction dicts with post-processed captions."""
    out = []
    for r in rows:
        item = dict(r)
        item["caption"] = postprocess_caption(r["caption"])
        out.append(item)
    return out


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_questions(questions, reference, split):
    if len(questions) != COUNTS[split]:
        raise ValueError(f"{split}: expected {COUNTS[split]} pairs, got {len(questions)}")
    ids = [q["id"] for q in questions]
    ref_ids = [im["id"] for im in reference["images"]]
    ann_ids = {a["image_id"] for a in reference["annotations"]}
    if len(set(ids)) != len(ids) or len(set(ref_ids)) != len(ref_ids):
        raise ValueError("Duplicate question or reference IDs")
    if set(ids) != set(ref_ids) or set(ids) != ann_ids:
        raise ValueError("Question/reference IDs do not match")
    for q in questions:
        if len(q["image"]) != 2 or q["conversations"][0]["value"].count("<image>") != 2:
            raise ValueError(f"Not an ordered two-image prompt: {q['id']}")
        for name in q["image"]:
            p = Path(name)
            if p.is_absolute() or ".." in p.parts or not name.startswith(f"{split}/test/images/"):
                raise ValueError(f"Unexpected test image path: {name}")
    return ids


def validate_predictions(rows, expected_ids):
    ids = [r["image_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate prediction IDs")
    missing, extra = set(expected_ids) - set(ids), set(ids) - set(expected_ids)
    if missing or extra or len(ids) != len(expected_ids):
        raise ValueError(f"Prediction coverage mismatch: missing={len(missing)}, extra={len(extra)}")
    if any(not isinstance(r.get("caption"), str) or not r["caption"].strip() for r in rows):
        raise ValueError("Empty or invalid caption")
    return True


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def score_table3(reference_file, prediction_file):
    """Same PTB tokenizer and four scorers as official COCOEvalCap; omit SPICE.

    Table 3 reports only BLEU-4/METEOR/ROUGE-L/CIDEr. Never score a partial
    prediction file against full-test paper values. COCO's CIDEr includes
    its own normalization: multiply the returned value by 100 only once.
    """
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider

    reference = read_json(reference_file)
    rows = read_jsonl(prediction_file)
    ids = [i["id"] for i in reference["images"]]
    validate_predictions(rows, ids)
    gts = {i: [] for i in ids}
    for a in reference["annotations"]:
        gts[a["image_id"]].append({"caption": a["caption"]})
    res = {r["image_id"]: [{"caption": r["caption"]}] for r in rows}
    tokenizer = PTBTokenizer()
    gts, res = tokenizer.tokenize(gts), tokenizer.tokenize(res)
    b, _ = Bleu(4).compute_score(gts, res)
    m, _ = Meteor().compute_score(gts, res)
    r, _ = Rouge().compute_score(gts, res)
    c, _ = Cider().compute_score(gts, res)
    return {"BLEU-4": float(b[3]), "METEOR": float(m), "ROUGE-L": float(r), "CIDEr": float(c)}
