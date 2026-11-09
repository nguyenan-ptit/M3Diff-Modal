"""Reproduce and improve OmniDiff Table 3 evaluation using the authors' merged checkpoint.

Stages:
  prepare         - CPU downloads of pinned model, vision tower, and test images
  smoke           - 10-pair sanity check on GPU
  full            - Original baseline evaluation (803 Real + 767 Render pairs)
  improve-post    - Fast CPU-only ablation: apply anti-loop & lexical post-processing
                    to an existing baseline run (e.g. full-20261101T074459Z-253021)
  improve-full    - Full GPU improved pipeline (Residual Differential Injection in MDP
                    + Calibrated Beam/Anti-Repetition Decoding + Post-processing),
                    running Real and Render in parallel across 2 GPUs.
  collect         - Score a completed run on CPU and download the result ZIP.
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
# Embedded so importing this module remotely needs no local JSON sidecar.
LOCK = {
    "repository": "https://github.com/Yuan-Liu-01/M3Diff.git",
    "code_commit": "4e346e52185620ec6fa17235d93b0c26a9d47a71",
    "model": {"repo_id": "IVC-liuyuan/M3Diff", "revision": "d6a8880dc789b0816660278c087ac9ebdfe18dd0"},
    "dataset": {"repo_id": "IVC-liuyuan/OmniDiff", "revision": "ed4a84cc51912c73e3d54c8643256d9469657552"},
    "vision": {"repo_id": "google/siglip-so400m-patch14-384", "revision": "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"},
}
COMMIT = LOCK["code_commit"]
CACHE = "/cache"
REPO = "/opt/M3Diff"
VOLUME_NAME = "omnidiff-reproduction-v1"
GPU = "A100-40GB"
app = modal.App("omnidiff-reproduction")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)


def add_source(image):
    return image.run_commands(
        f"git clone {LOCK['repository']} {REPO}",
        f"git -C {REPO} checkout --detach {COMMIT}",
    ).workdir(REPO)


def add_helpers(image):
    return image.add_local_file(HERE / "repro_helpers.py", f"{REPO}/repro_helpers.py")


cpu_image = add_helpers(add_source(
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "default-jre-headless")
    .pip_install(
        "huggingface-hub==0.36.0", "Pillow==10.4.0", "numpy==1.26.4",
        "pycocoevalcap==1.2", "pycocotools==2.0.10",
    )
).env({"HF_HOME": "/cache/hf", "HF_HUB_DISABLE_TELEMETRY": "1"}))

gpu_image = add_helpers(add_source(
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential")
    .pip_install("setuptools==75.8.0", "wheel==0.45.1", "packaging==24.2", "ninja==1.11.1.3")
    .pip_install("torch==2.2.2", "torchvision==0.17.2", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("numpy==1.26.4", "accelerate==0.29.1", "Pillow==10.4.0", "huggingface-hub==0.36.0")
    .env({"MAX_JOBS": "4", "TORCH_CUDA_ARCH_LIST": "8.0"})
    .run_commands("pip install flash-attn==2.5.8 --no-build-isolation")
).run_commands(
    f"pip install -e {REPO}",
    "pip check",
).env({
    "HF_HOME": "/cache/hf", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false", "PYTHONHASHSEED": "0",
    "PYTHONPATH": REPO, "PYTHONUNBUFFERED": "1",
}))


@app.function(image=cpu_image, volumes={CACHE: volume}, cpu=2, memory=8192,
              timeout=3600, max_containers=1, retries=0, scaledown_window=2)
def prepare():
    """Download pinned assets on CPU before reserving GPU time."""
    import sys
    from huggingface_hub import snapshot_download
    from PIL import Image
    sys.path.insert(0, REPO)
    from repro_helpers import read_json, write_json, validate_questions, sha256

    volume.reload()
    root = Path(CACHE)
    manifest_path = root / "prepared.json"
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if previous.get("lock") == LOCK:
            print("Pinned assets already prepared.")
            return previous
        raise RuntimeError("Volume contains another asset revision. Use a new VOLUME_NAME.")

    data_root = root / "datasets/OmniDiff"
    snapshot_download(**LOCK["dataset"], repo_type="dataset", local_dir=str(data_root),
                      allow_patterns=["Real/test/images/*", "Render/test/images/*", "LICENSE", "README.md"],
                      max_workers=4)
    counts, hashes, image_hashes = {}, {}, {}
    for split in ("Real", "Render"):
        qpath = Path(REPO) / f"data/evaluation/prompt/OmniDiff_{split}_test.json"
        rpath = Path(REPO) / f"data/evaluation/reference/OmniDiff_{split}.json"
        questions, reference = read_json(qpath), read_json(rpath)
        validate_questions(questions, reference, split)
        counts[split] = len(questions)
        hashes[split] = {"prompt": sha256(qpath), "reference": sha256(rpath)}
        for name in sorted({n for q in questions for n in q["image"]}):
            path = data_root / name
            with Image.open(path) as im:
                im.verify()
            image_hashes[name] = sha256(path)

    model_source = Path(snapshot_download(**LOCK["model"],
        allow_patterns=["*.json", "*.txt", "*.safetensors", "LICENSE", "README.md"], max_workers=2))
    vision_source = snapshot_download(**LOCK["vision"],
        allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json"], max_workers=2)
    config = read_json(model_source / "config.json")
    index = read_json(model_source / "model.safetensors.index.json")
    assert config["mm_resampler_type"] == "mdp"
    selected = config["mm_vision_select_layer"]
    assert (json.loads(selected) if isinstance(selected, str) else selected) == [-11, -8, -5, -2]
    assert config.get("unfreeze_mm_vision_tower") or "mm_vision_tower" in config.get("mm_tunable_parts", "")
    assert any(k.startswith("model.vision_tower.") for k in index["weight_map"])
    assert any("vision_resampler" in k for k in index["weight_map"])
    for filename in set(index["weight_map"].values()):
        if not (model_source / filename).is_file():
            raise FileNotFoundError(filename)

    # The official loader recognizes multimodal models by directory basename.
    # Only the external vision path changes; learned weights/config remain intact.
    model_dir = root / "models/m3diff"
    model_dir.mkdir(parents=True, exist_ok=True)
    for src in model_source.iterdir():
        dst = model_dir / src.name
        if src.name != "config.json" and src.is_file() and not dst.exists():
            dst.symlink_to(src)
    original_vision = config["mm_vision_tower"]
    config["mm_vision_tower"] = vision_source
    write_json(model_dir / "config.json", config)
    manifest = {
        "lock": LOCK, "counts": counts, "file_sha256": hashes,
        "image_sha256": image_hashes, "model_path": str(model_dir),
        "image_root": str(data_root), "vision_source": vision_source,
        "original_mm_vision_tower": original_vision,
        "original_config_sha256": sha256(model_source / "config.json"),
        "runtime_config_sha256": sha256(model_dir / "config.json"),
        "model_index_sha256": sha256(model_source / "model.safetensors.index.json"),
        "model_size_bytes": index["metadata"]["total_size"],
        "scope": "released-checkpoint evaluation; no training",
    }
    write_json(manifest_path, manifest)
    volume.commit()
    print(json.dumps({"prepared": True, "pairs": counts, "unique_images": len(image_hashes)}, indent=2))
    return manifest


@app.function(image=gpu_image, gpu=GPU, volumes={CACHE: volume}, cpu=4, memory=32768,
              timeout=7200, max_containers=1, retries=0, scaledown_window=2)
def infer(stage: str, run_id: str):
    """Call the official evaluator; retain stdout/stderr and predictions."""
    import datetime
    import random
    import subprocess
    import sys
    import time
    import traceback
    sys.path.insert(0, REPO)
    from repro_helpers import read_json, write_json, read_jsonl, validate_predictions, sha256

    if stage not in {"smoke", "full"}:
        raise ValueError("stage must be smoke or full")
    volume.reload()
    prepared = read_json(Path(CACHE) / "prepared.json")
    if prepared["lock"] != LOCK:
        raise ValueError("Asset revisions differ; run prepare with matching lock")
    run_dir = Path(CACHE) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "assets.json", prepared)
    with (run_dir / "pip-freeze.txt").open("w", encoding="utf-8") as versions:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, stdout=versions)
    gpu_info = subprocess.check_output(["nvidia-smi"], text=True)
    (run_dir / "nvidia-smi.txt").write_text(gpu_info, encoding="utf-8")
    report = {"run_id": run_id, "stage": stage, "gpu_request": GPU,
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "status": "running", "scope": "released-checkpoint evaluation",
              "decoding": {"dtype": "bfloat16", "flash_attention": True,
                           "temperature": 0, "num_beams": 1, "max_new_tokens": 1024,
                           "conv_mode": "qwen_2", "use_cache": True, "extra_prompt": ""},
              "splits": {}}
    write_json(run_dir / "run.json", report)
    volume.commit()
    try:
        for split in ("Real", "Render"):
            qpath = Path(REPO) / f"data/evaluation/prompt/OmniDiff_{split}_test.json"
            rpath = Path(REPO) / f"data/evaluation/reference/OmniDiff_{split}.json"
            for label, path in (("prompt", qpath), ("reference", rpath)):
                if sha256(path) != prepared["file_sha256"][split][label]:
                    raise ValueError(f"{split} {label} file changed")
            questions = read_json(qpath)
            if stage == "smoke":
                # Fixed preselected sample; never select by score after inference.
                questions = random.Random(2025).sample(questions, 5)
            split_dir = run_dir / split
            split_dir.mkdir()
            selected = split_dir / "questions.json"
            write_json(selected, questions)
            prediction = split_dir / "predictions.jsonl"
            command = [
                sys.executable, f"{REPO}/llava/eval/model_vqa_qwen_OmniDiff_{split}.py",
                "--model-path", prepared["model_path"],
                "--image-folder", prepared["image_root"],
                "--question-file", str(selected), "--answers-file", str(prediction),
                "--conv-mode", "qwen_2", "--num-chunks", "1", "--chunk-idx", "0",
                "--temperature", "0", "--num_beams", "1", "--max-new-tokens", "1024",
                "--torch-dtype", "bfloat16", "--flash-attention", "--use-cache",
            ]
            write_json(split_dir / "command.json", command)
            start = time.monotonic()
            # Unmodified upstream inference runs in a fresh process per split.
            with (split_dir / "inference.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
                if code:
                    raise RuntimeError(f"{split} inference exited {code}; inspect inference.log")
            rows = read_jsonl(prediction)
            validate_predictions(rows, [q["id"] for q in questions])
            report["splits"][split] = {"pairs": len(rows), "elapsed_seconds": time.monotonic() - start,
                                       "predictions_sha256": sha256(prediction)}
            ref = read_json(rpath)
            ref_captions = {a["image_id"]: a["caption"] for a in ref["annotations"]}
            preds = {r["image_id"]: r["caption"] for r in rows}
            write_json(split_dir / "examples.json", [
                {"id": q["id"], "image": q["image"], "prompt": q["conversations"][0]["value"],
                 "prediction": preds[q["id"]], "reference": ref_captions[q["id"]]}
                for q in questions
            ])
            write_json(run_dir / "run.json", report)
            volume.commit()
        report["status"] = "inference_complete"
        return run_id
    except Exception:
        report["status"] = "failed"
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        write_json(run_dir / "run.json", report)
        volume.commit()


@app.function(image=gpu_image, gpu=GPU, volumes={CACHE: volume}, cpu=4, memory=32768,
              timeout=900, max_containers=1, retries=0, scaledown_window=2)
def smoke(run_id: str):
    """Run the same inference function with a shorter 15-minute timeout."""
    return infer.local("smoke", run_id)


@app.function(image=gpu_image, gpu=GPU, volumes={CACHE: volume}, cpu=4, memory=32768,
              timeout=7200, max_containers=2, retries=0, scaledown_window=2)
def infer_improved_split(
    split: str,
    run_id: str,
    num_beams: int = 3,
    repetition_penalty: float = 1.12,
    length_penalty: float = 1.05,
    no_repeat_ngram_size: int = 4,
    mdp_beta: float = 0.08,
):
    """Run improved M3Diff inference on a single split with:
    1. Residual Differential Injection in MDP (injecting +/- beta * change_fused)
    2. Anti-Repetition & Length-Calibrated Beam Search
    3. Post-processing against degenerate loops and punctuation boundary bugs.
    """
    import os
    import re
    import sys
    import time
    import torch
    from PIL import Image
    from tqdm import tqdm

    sys.path.insert(0, REPO)
    from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images
    from llava.model.builder import load_pretrained_model
    from llava.model.multimodal_resampler.mdp import MDP
    from llava.utils import disable_torch_init
    from repro_helpers import (
        postprocess_predictions,
        read_json,
        sha256,
        validate_predictions,
        write_json,
        write_jsonl,
    )

    volume.reload()
    prepared = read_json(Path(CACHE) / "prepared.json")
    run_dir = Path(CACHE) / "runs" / run_id
    split_dir = run_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    qpath = Path(REPO) / f"data/evaluation/prompt/OmniDiff_{split}_test.json"
    rpath = Path(REPO) / f"data/evaluation/reference/OmniDiff_{split}.json"
    questions = read_json(qpath)
    write_json(split_dir / "questions.json", questions)

    # 1. Patch MDP.forward to inject residual differential representation (change_fused)
    orig_mdp_forward = MDP.forward

    def improved_mdp_forward(self, imageA_emb: list, imageB_emb: list):
        imageA_fused, imageB_fused, change_fused = orig_mdp_forward(self, imageA_emb, imageB_emb)
        if mdp_beta != 0.0:
            # Contrastive differential injection: highlight directional change A -> B
            imageA_fused = imageA_fused - mdp_beta * change_fused
            imageB_fused = imageB_fused + mdp_beta * change_fused
        return imageA_fused, imageB_fused, change_fused

    MDP.forward = improved_mdp_forward

    start = time.monotonic()
    disable_torch_init()
    model_path = prepared["model_path"]
    model_name = get_model_name_from_path(model_path)
    torch_dtype = torch.bfloat16
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, model_name, use_flash_attn=True, torch_dtype=torch_dtype
    )

    from llava.eval.model_vqa_qwen_OmniDiff_Real import preprocess_qwen

    raw_rows = []
    image_folder = prepared["image_root"]
    for line in tqdm(questions, desc=f"Improved {split}"):
        image_files = line["image"]
        qs = line["conversations"][0]["value"]
        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        input_ids = preprocess_qwen(
            [{"from": "human", "value": qs}, {"from": "gpt", "value": None}],
            tokenizer,
            has_image=True,
        ).cuda()
        image_tensors = []
        for image_file in image_files:
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
            image_tensors.append(
                process_images([image], image_processor, model.config).to(torch_dtype)
            )
        image_tensors = [torch.cat(image_tensors, dim=0)]
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                inputs=input_ids,
                images=image_tensors,
                do_sample=False,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_new_tokens=512,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)].strip()
        raw_rows.append({"id": line["id"], "image_id": line["id"], "caption": outputs})

    raw_pred_path = split_dir / "predictions_raw.jsonl"
    write_jsonl(raw_pred_path, raw_rows)

    # Apply post-processing
    cleaned_rows = postprocess_predictions(raw_rows)
    pred_path = split_dir / "predictions.jsonl"
    write_jsonl(pred_path, cleaned_rows)
    validate_predictions(cleaned_rows, [q["id"] for q in questions])

    ref = read_json(rpath)
    ref_captions = {a["image_id"]: a["caption"] for a in ref["annotations"]}
    preds = {r["image_id"]: r["caption"] for r in cleaned_rows}
    raw_preds = {r["image_id"]: r["caption"] for r in raw_rows}
    write_json(
        split_dir / "examples.json",
        [
            {
                "id": q["id"],
                "image": q["image"],
                "prompt": q["conversations"][0]["value"],
                "prediction": preds[q["id"]],
                "prediction_raw": raw_preds[q["id"]],
                "reference": ref_captions[q["id"]],
            }
            for q in questions
        ],
    )
    elapsed = time.monotonic() - start
    split_meta = {
        "pairs": len(cleaned_rows),
        "elapsed_seconds": elapsed,
        "predictions_sha256": sha256(pred_path),
    }
    write_json(split_dir / "split_meta.json", split_meta)
    volume.commit()
    return split_meta


@app.function(image=cpu_image, volumes={CACHE: volume}, cpu=2, memory=8192,
              timeout=1800, max_containers=1, retries=0, scaledown_window=2)
def improve_from_existing(base_run_id: str, new_run_id: str):
    """Apply post-processing improvements to an existing completed run on CPU and score."""
    import datetime
    import sys

    sys.path.insert(0, REPO)
    from repro_helpers import (
        postprocess_predictions,
        read_json,
        read_jsonl,
        sha256,
        validate_predictions,
        write_json,
        write_jsonl,
    )

    volume.reload()
    base_dir = Path(CACHE) / "runs" / base_run_id
    if not base_dir.exists():
        raise ValueError(f"Base run directory not found on volume: {base_run_id}")
    run_dir = Path(CACHE) / "runs" / new_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    prepared = read_json(Path(CACHE) / "prepared.json")
    write_json(run_dir / "assets.json", prepared)

    report = {
        "run_id": new_run_id,
        "base_run_id": base_run_id,
        "stage": "full",
        "method": "M3Diff + Anti-Loop & Lexical Post-Processing",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "running",
        "splits": {},
    }
    for split in ("Real", "Render"):
        split_dir = run_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        qpath = Path(REPO) / f"data/evaluation/prompt/OmniDiff_{split}_test.json"
        rpath = Path(REPO) / f"data/evaluation/reference/OmniDiff_{split}.json"
        questions = read_json(qpath)
        raw_rows = read_jsonl(base_dir / split / "predictions.jsonl")
        cleaned_rows = postprocess_predictions(raw_rows)
        pred_path = split_dir / "predictions.jsonl"
        write_jsonl(pred_path, cleaned_rows)
        validate_predictions(cleaned_rows, [q["id"] for q in questions])

        ref = read_json(rpath)
        ref_captions = {a["image_id"]: a["caption"] for a in ref["annotations"]}
        preds = {r["image_id"]: r["caption"] for r in cleaned_rows}
        raw_preds = {r["image_id"]: r["caption"] for r in raw_rows}
        changed_count = sum(1 for k in preds if preds[k] != raw_preds[k])
        write_json(
            split_dir / "examples.json",
            [
                {
                    "id": q["id"],
                    "image": q["image"],
                    "prompt": q["conversations"][0]["value"],
                    "prediction": preds[q["id"]],
                    "prediction_before_postprocess": raw_preds[q["id"]],
                    "reference": ref_captions[q["id"]],
                }
                for q in questions
            ],
        )
        report["splits"][split] = {
            "pairs": len(cleaned_rows),
            "modified_captions": changed_count,
            "predictions_sha256": sha256(pred_path),
        }
    report["status"] = "inference_complete"
    write_json(run_dir / "run.json", report)
    volume.commit()
    return new_run_id


@app.function(image=cpu_image, volumes={CACHE: volume}, cpu=2, memory=8192,
              timeout=600, max_containers=1, retries=0, scaledown_window=2)
def init_improved_run(run_id: str, config: dict):
    """Initialize run metadata directory on the shared volume before parallel split execution."""
    import datetime
    import sys

    sys.path.insert(0, REPO)
    from repro_helpers import read_json, write_json

    volume.reload()
    prepared = read_json(Path(CACHE) / "prepared.json")
    run_dir = Path(CACHE) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "assets.json", prepared)
    report = {
        "run_id": run_id,
        "stage": "full",
        "gpu_request": f"2x {GPU} (parallel splits)",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "running",
        "scope": "improved-inference evaluation",
        "improvements": config,
        "splits": {},
    }
    write_json(run_dir / "run.json", report)
    volume.commit()


@app.function(image=cpu_image, volumes={CACHE: volume}, cpu=2, memory=8192,
              timeout=600, max_containers=1, retries=0, scaledown_window=2)
def finalize_improved_run(run_id: str, splits_meta: dict):
    """Mark parallel split inference as complete."""
    import sys

    sys.path.insert(0, REPO)
    from repro_helpers import read_json, write_json

    volume.reload()
    run_dir = Path(CACHE) / "runs" / run_id
    report = read_json(run_dir / "run.json")
    report["splits"] = splits_meta
    report["status"] = "inference_complete"
    write_json(run_dir / "run.json", report)
    volume.commit()


@app.function(image=cpu_image, volumes={CACHE: volume}, cpu=2, memory=8192,
              timeout=1800, max_containers=1, retries=0, scaledown_window=2)
def collect(run_id: str):
    """Score complete runs on CPU and return a small result archive."""
    import csv
    import io
    import subprocess
    import sys
    import zipfile
    sys.path.insert(0, REPO)
    from repro_helpers import read_json, write_json, score_table3, PAPER, BASELINE_REPRODUCED
    volume.reload()
    run_dir = Path(CACHE) / "runs" / run_id
    report = read_json(run_dir / "run.json")
    if report["status"] not in {"inference_complete", "complete"}:
        raise ValueError("Inference is incomplete; do not report partial results as full-test scores")
    with (run_dir / "scoring-pip-freeze.txt").open("w", encoding="utf-8") as versions:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, stdout=versions)
    if report["stage"] == "full":
        table = []
        for split in ("Real", "Render"):
            reference = Path(REPO) / f"data/evaluation/reference/OmniDiff_{split}.json"
            scores = score_table3(reference, run_dir / split / "predictions.jsonl")
            write_json(run_dir / split / "metrics_raw.json", scores)
            write_json(run_dir / split / "metrics_x100.json", {k: v * 100 for k, v in scores.items()})
            for metric, raw in scores.items():
                val_x100 = raw * 100
                base_val = BASELINE_REPRODUCED[split][metric]
                table.append({
                    "split": split,
                    "metric": metric,
                    "paper": PAPER[split][metric],
                    "baseline_reproduced": base_val,
                    "reproduced_x100": val_x100,
                    "difference": val_x100 - PAPER[split][metric],
                    "gain_vs_baseline": val_x100 - base_val,
                })
        with (run_dir / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        print(json.dumps(table, indent=2))
        report["status"] = "complete"
    else:
        (run_dir / "SMOKE_ONLY.txt").write_text(
            "10-pair execution check only. No benchmark score or training claim.\n", encoding="utf-8")
        report["status"] = "complete"
    write_json(run_dir / "run.json", report)
    volume.commit()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(run_dir))
    return buffer.getvalue()


@app.local_entrypoint()
def main(
    stage: str = "prepare",
    run_id: str = "",
    base_run_id: str = "full-20261101T074459Z-253021",
    num_beams: int = 3,
    repetition_penalty: float = 1.12,
    length_penalty: float = 1.05,
    no_repeat_ngram_size: int = 4,
    mdp_beta: float = 0.08,
):
    """--stage prepare|smoke|full|improve-post|improve-full|collect [--run-id ID]."""
    import datetime
    import re
    import uuid
    valid_stages = {"prepare", "smoke", "full", "improve-post", "improve-full", "collect"}
    if stage not in valid_stages:
        raise ValueError(f"Use one of: {', '.join(sorted(valid_stages))}")
    if stage == "prepare":
        result = prepare.remote()
        print("Prepared pairs:", result["counts"])
        print("Next: modal run modal_reproduce.py --stage smoke")
        return
    if stage == "collect" and not run_id:
        raise ValueError("collect requires --run-id from the previous run")
    if not run_id:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{stage}-{stamp}-{uuid.uuid4().hex[:6]}"
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
        raise ValueError("Invalid run ID")
    print("RUN ID:", run_id, flush=True)
    print("Modal volume:", VOLUME_NAME, flush=True)
    if stage == "smoke":
        smoke.remote(run_id)
    elif stage == "full":
        infer.remote(stage, run_id)
    elif stage == "improve-post":
        print(f"Applying anti-loop & lexical post-processing on base run: {base_run_id}", flush=True)
        improve_from_existing.remote(base_run_id, run_id)
    elif stage == "improve-full":
        config = {
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
            "length_penalty": length_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "mdp_beta": mdp_beta,
            "postprocess": True,
        }
        print("Running parallel GPU inference with improvements:", json.dumps(config), flush=True)
        init_improved_run.remote(run_id, config)
        real_call = infer_improved_split.spawn(
            "Real", run_id, num_beams, repetition_penalty, length_penalty, no_repeat_ngram_size, mdp_beta
        )
        render_call = infer_improved_split.spawn(
            "Render", run_id, num_beams, repetition_penalty, length_penalty, no_repeat_ngram_size, mdp_beta
        )
        splits_meta = {
            "Real": real_call.get(),
            "Render": render_call.get(),
        }
        finalize_improved_run.remote(run_id, splits_meta)
    data = collect.remote(run_id)
    results = HERE / "results"
    results.mkdir(exist_ok=True)
    archive_path = results / f"{run_id}.zip"
    archive_path.write_bytes(data)
    print("Saved:", archive_path)
