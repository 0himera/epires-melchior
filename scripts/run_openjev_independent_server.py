"""Run the frozen independent corpus against base and four selected pilot adapters.

Execute this on the GPU host after copying the corpus and evaluator to --benchmark.
Each model runs in a fresh, short-lived container; serving containers stay live.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


MODELS = ("base", "s42_c00", "s42_c01", "s43_c00", "s43_c01")


def main(args):
    benchmark = args.benchmark.resolve()
    (benchmark / "scores").mkdir(exist_ok=True)
    corpus_hash = hashlib.sha256((benchmark / "holdout.jsonl").read_bytes()).hexdigest()
    evaluator_hash = hashlib.sha256((benchmark / "evaluate_openjev_pilot_holdout.py").read_bytes()).hexdigest()
    for name in MODELS:
        output = benchmark / "scores" / f"{name}.json"
        if output.exists():
            existing = json.loads(output.read_text())
            if existing["data_sha256"]["/benchmark/holdout.jsonl"] != corpus_hash:
                raise ValueError(f"Frozen corpus changed since {name} was scored")
            continue
        command = ["docker", "run", "--rm", "--name", f"openjev-independent-{name}",
                   "--device", "/dev/kfd", "--device", "/dev/dri", "--group-add", "video",
                   "--ipc", "host", "--security-opt", "seccomp=unconfined", "--network", "none",
                   "-e", "PYTHONPATH=/repo", "-e", "OMP_NUM_THREADS=4",
                   "-e", "TOKENIZERS_PARALLELISM=false", "-e", "HF_HUB_OFFLINE=1",
                   "-v", f"{args.repository.resolve()}:/repo:ro",
                   "-v", f"{args.model_root.resolve()}:/model:ro",
                   "-v", f"{args.experiment.resolve()}:/experiment:ro",
                   "-v", f"{benchmark}:/benchmark",
                   "--entrypoint", "python3", args.image, "-u",
                   "/benchmark/evaluate_openjev_pilot_holdout.py",
                   "--model", "/model/qwen3.5-4b-nli-v5",
                   "--datasets", "/benchmark/holdout.jsonl",
                   "--output", f"/benchmark/scores/{name}.json",
                   "--scope", "independent frozen portfolio benchmark; checkpoint choice fixed on development data"]
        if name != "base":
            selected = json.loads((args.experiment / "runs" / name / "summary.json").read_text())["selected_epoch"]
            command += ["--adapter", f"/experiment/runs/{name}/epoch_{selected}"]
        started = time.time()
        with (benchmark / "scores" / f"{name}.log").open("w") as log:
            print(f"Scoring {name}; corpus SHA-256 {corpus_hash}", flush=True)
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Scored {name} in {time.time()-started:.1f}s", flush=True)
    manifest = {"schema": "independent-score-manifest-v1", "corpus_sha256": corpus_hash,
                "evaluator_sha256": evaluator_hash, "image": args.image,
                "models": {name: {"score_sha256": hashlib.sha256((benchmark / "scores" / f"{name}.json").read_bytes()).hexdigest(),
                                  "summary": json.loads((benchmark / "scores" / f"{name}.json").read_text())["aggregate"]}
                           for name in MODELS}}
    (benchmark / "scores" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["models"], indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=Path, default=Path("/data/openjev_independent_20260929"))
    p.add_argument("--repository", type=Path, default=Path("/data/epires-melchior"))
    p.add_argument("--model-root", type=Path, default=Path("/data/models/openjev"))
    p.add_argument("--experiment", type=Path, default=Path("/data/openjev_adapt_20260929"))
    p.add_argument("--image", default="sha256:30761c2125ce150d556bef46406a0158446421886bf83a2e60154c6e4ca17a13")
    main(p.parse_args())
