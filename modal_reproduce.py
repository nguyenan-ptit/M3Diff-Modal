"""Reproduce OmniDiff Table 3 evaluation using the authors' merged checkpoint.

Stages: prepare (CPU downloads), smoke (10 pairs), full (803 + 767 pairs).
No remote work happens on import or with --help. Running a stage uses Modal
compute and can incur charges. See README_vi.md before launching.
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
    from repro_helpers import read_json, write_json, score_table3, PAPER
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
                table.append({"split": split, "metric": metric, "paper": PAPER[split][metric],
                              "reproduced_x100": raw * 100, "difference": raw * 100 - PAPER[split][metric]})
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
def main(stage: str = "prepare", run_id: str = ""):
    """--stage prepare|smoke|full|collect [--run-id ID]."""
    import datetime
    import re
    import uuid
    if stage not in {"prepare", "smoke", "full", "collect"}:
        raise ValueError("Use prepare, smoke, full or collect")
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
    data = collect.remote(run_id)
    results = HERE / "results"
    results.mkdir(exist_ok=True)
    archive_path = results / f"{run_id}.zip"
    archive_path.write_bytes(data)
    print("Saved:", archive_path)
