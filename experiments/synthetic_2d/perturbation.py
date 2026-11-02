#!/usr/bin/env python3
"""Paired 2D synthetic perturbation experiment reusing an alpha panel.

For every saved alpha-panel fit on D, replace the first observation by the
fixed point (3, 3), refit only D', and compare the paired private and
non-private outputs.  The private D' fit uses the same LMC, histogram, and
reverse-diffusion random streams as the saved D fit.  This isolates the
response to the one-record replacement while avoiding a second computation of
the baseline D release.

The companion ``two_dimensional_complete_experiment.py`` and the alpha-panel
output must be available.  Formal defaults are inherited from its paper
profile: gamma=0.5, n in {1000,2000,4000,8000,16000}, score alpha in
{0.5,1,2,4,8}, 100 repetitions, 500,000 projected-LMC steps, and 20,000
reverse particles.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np


FIXED_POINT = np.array([3.0, 3.0], dtype=float)
SCHEMA = 1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_engine(path: Path):
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing companion experiment file: {path}")
    name = "two_dimensional_complete_experiment_reused"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tag_for(engine, n: int, rep: int, alpha: float, gamma: float) -> str:
    return f"n{n}_r{rep:03d}_g{gamma:g}_a{alpha:g}".replace(".", "p")


def find_base(alpha_roots: list[Path], tag: str) -> tuple[Path, Path]:
    matches = []
    for root in alpha_roots:
        artifact = root / "alpha" / "artifacts" / f"{tag}.npz"
        row = root / "alpha" / "rows" / f"{tag}.json"
        if artifact.exists() and row.exists():
            matches.append((artifact, row))
    if not matches:
        roots = ", ".join(str(p) for p in alpha_roots)
        raise FileNotFoundError(f"Missing alpha-panel artifact {tag} under: {roots}")
    if len(matches) > 1:
        hashes = {(sha256(a), sha256(r)) for a, r in matches}
        if len(hashes) != 1:
            raise RuntimeError(f"Conflicting copies of alpha-panel artifact {tag}")
    return matches[0]


def load_base(artifact: Path, row_path: Path):
    with np.load(artifact, allow_pickle=False) as saved:
        required = {
            "data", "private_theta", "private_samples", "private_edges",
            "private_mass", "private_slope", "nonprivate_theta",
            "nonprivate_samples", "nonprivate_edges", "nonprivate_mass",
            "nonprivate_slope", "class_json",
        }
        missing = required.difference(saved.files)
        if missing:
            raise RuntimeError(f"{artifact} is missing {sorted(missing)}")
        arrays = {key: saved[key].copy() for key in required if key != "class_json"}
        class_json = json.loads(str(saved["class_json"]))
    rows = json.loads(row_path.read_text())
    if not isinstance(rows, list):
        raise RuntimeError(f"Expected a list in {row_path}")
    by_method = {row["method"]: row for row in rows}
    if set(("private", "nonprivate")).difference(by_method):
        raise RuntimeError(f"Missing private or nonprivate row in {row_path}")
    return arrays, class_json, by_method


def make_fit(arrays, method: str, info: dict) -> dict:
    return {
        "theta": arrays[f"{method}_theta"],
        "y": arrays[f"{method}_samples"],
        "cells": {
            "edges": arrays[f"{method}_edges"],
            "mass": arrays[f"{method}_mass"],
            "slope": arrays[f"{method}_slope"],
        },
        "info": dict(info),
    }


def config_from_args(engine, args):
    config = engine.profile(args.profile)
    overrides = {}
    if args.n_values is not None:
        overrides["n_values"] = tuple(args.n_values)
    if args.score_alphas is not None:
        overrides["score_alphas"] = tuple(args.score_alphas)
    if args.repetitions is not None:
        overrides["repetitions"] = args.repetitions
    if args.lmc_steps is not None:
        overrides["lmc_steps"] = args.lmc_steps
    if args.particles is not None:
        overrides["particles"] = args.particles
    if overrides:
        from dataclasses import replace
        config = replace(config, **overrides)
    return config


def validate_base_config(engine, config, class_json: dict, n: int, alpha: float):
    expected = engine.make_class(n, alpha, config.sensitivity_gamma, config)
    for key in ("n", "gamma", "R", "b", "b_raw", "jmin", "m", "B", "D"):
        observed = class_json[key]
        target = expected[key]
        if isinstance(target, float):
            if not math.isclose(float(observed), target, rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError(f"Base class mismatch for {key}: {observed} != {target}")
        elif observed != target:
            raise RuntimeError(f"Base class mismatch for {key}: {observed} != {target}")
    if np.max(np.abs(FIXED_POINT)) > expected["R"]:
        raise RuntimeError(
            f"Fixed replacement {FIXED_POINT.tolist()} lies outside R={expected['R']}"
        )
    return expected


def run_one(payload):
    (source, alpha_roots_raw, output_raw, profile, n_values, score_alphas,
     repetitions, lmc_steps, particles, n, rep, alpha) = payload
    source = Path(source)
    alpha_roots = [Path(p) for p in alpha_roots_raw]
    output = Path(output_raw)
    engine = load_engine(source)
    namespace = argparse.Namespace(
        profile=profile, n_values=n_values, score_alphas=score_alphas,
        repetitions=repetitions, lmc_steps=lmc_steps, particles=particles,
    )
    config = config_from_args(engine, namespace)
    gamma = config.sensitivity_gamma
    tag = tag_for(engine, n, rep, alpha, gamma)
    destination = output / "rows" / f"{tag}.json"
    artifact_out = output / "artifacts" / f"{tag}.npz"
    if destination.exists() and artifact_out.exists():
        return json.loads(destination.read_text())

    started = time.perf_counter()
    base_artifact, base_rows = find_base(alpha_roots, tag)
    arrays, class_json, base_info = load_base(base_artifact, base_rows)
    expected_class = validate_base_config(engine, config, class_json, n, alpha)
    data = arrays["data"]
    if data.shape != (n, 2):
        raise RuntimeError(f"Unexpected data shape {data.shape} in {base_artifact}")
    if engine.hash_array(data) != base_info["private"]["train_hash"]:
        raise RuntimeError(f"Training-data hash mismatch in {base_artifact}")

    neighbor_data = data.copy()
    old_value = neighbor_data[0].copy()
    neighbor_data[0] = FIXED_POINT
    if np.count_nonzero(np.any(data != neighbor_data, axis=1)) != 1:
        raise RuntimeError("The neighboring datasets must differ in exactly one row")

    model = engine.fit_statistics(data, expected_class)
    neighbor_model = engine.fit_statistics(neighbor_data, expected_class)
    test = engine.draw(
        config.test_particles,
        engine.rng(config, rep, round(gamma * 1000), 2),
        config,
        gamma,
    )
    gcode = round(gamma * 1000)
    acode = round(alpha * 1000)
    private_seeds = [
        engine.stream_seed(config, n, rep, gcode, acode, 1, key)
        for key in (10, 11, 12)
    ]
    nonprivate_seeds = [
        engine.stream_seed(config, n, rep, gcode, 0, 0, key)
        for key in (10, 11, 12)
    ]
    checkpoint_dir = output / "checkpoints" / tag
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    base_private = make_fit(arrays, "private", base_info["private"])
    base_nonprivate = make_fit(arrays, "nonprivate", base_info["nonprivate"])
    neighbor_private = engine.fit_pipeline(
        neighbor_data, neighbor_model, alpha, private_seeds, True, config,
        checkpoint_dir, "private_neighbor_fixed3",
    )
    neighbor_nonprivate = engine.fit_pipeline(
        neighbor_data, neighbor_model, alpha, nonprivate_seeds, False, config,
        checkpoint_dir, "nonprivate_neighbor_fixed3",
    )

    comparisons = []
    for mode, first, second, metric_key in (
        ("private_fixed3_crn", base_private, neighbor_private, 210),
        ("nonprivate_fixed3_crn", base_nonprivate, neighbor_nonprivate, 211),
    ):
        metrics = engine.sensitivity_metrics(
            first, second, model, test, config,
            engine.stream_seed(config, n, rep, acode, metric_key),
        )
        comparisons.append({
            "task": "fixed3_reused_alpha_perturbation",
            "mode": mode,
            "n": n,
            "rep": rep,
            "gamma": gamma,
            "alpha_score": alpha,
            "alpha_total": 2 * alpha,
            "replacement_index": 0,
            "old_x1": float(old_value[0]),
            "old_x2": float(old_value[1]),
            "new_x1": 3.0,
            "new_x2": 3.0,
            "train_hash": engine.hash_array(data),
            "neighbor_hash": engine.hash_array(neighbor_data),
            "base_artifact": str(base_artifact.resolve()),
            "base_artifact_sha256": sha256(base_artifact),
            "base_release_reused": True,
            "coefficient_count": 2 * expected_class["m"] ** 2,
            "N": first["info"]["N"],
            "particles": config.particles,
            **metrics,
            "job_seconds_this_invocation": time.perf_counter() - started,
        })

    atomic_npz(
        artifact_out,
        replaced_index=np.array(0),
        old_value=old_value,
        fixed_value=FIXED_POINT,
        train_hash=np.array(engine.hash_array(data)),
        neighbor_hash=np.array(engine.hash_array(neighbor_data)),
        private_neighbor_theta=neighbor_private["theta"],
        private_neighbor_edges=neighbor_private["cells"]["edges"],
        private_neighbor_mass=neighbor_private["cells"]["mass"],
        private_neighbor_slope=neighbor_private["cells"]["slope"],
        nonprivate_neighbor_theta=neighbor_nonprivate["theta"],
        nonprivate_neighbor_edges=neighbor_nonprivate["cells"]["edges"],
        nonprivate_neighbor_mass=neighbor_nonprivate["cells"]["mass"],
        nonprivate_neighbor_slope=neighbor_nonprivate["cells"]["slope"],
        base_artifact=np.array(str(base_artifact.resolve())),
        base_artifact_sha256=np.array(sha256(base_artifact)),
    )
    atomic_json(destination, comparisons)
    # Completed rows and compact releases are sufficient to reproduce every
    # reported metric.  Remove large chain/reverse checkpoints only after both
    # final files are durable; interrupted jobs retain checkpoints and resume.
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    return comparisons


def manifest_for(source: Path, alpha_roots: list[Path], config) -> dict:
    from dataclasses import asdict
    return {
        "schema": SCHEMA,
        "experiment": "2d_synthetic_fixed_first_record_to_3_3_reusing_alpha_panel",
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "alpha_outputs": [str(p.resolve()) for p in alpha_roots],
        "fixed_replacement_index": 0,
        "fixed_replacement_point": [3.0, 3.0],
        "comparisons": ["private_fixed3_crn", "nonprivate_fixed3_crn"],
        "base_release_reused": True,
        "config": asdict(config),
    }


def ensure_manifest(path: Path, manifest: dict) -> None:
    import fcntl

    # Normalize tuples exactly as they appear after a JSON round trip.  This
    # keeps restarts and concurrent array tasks from reporting a false mismatch.
    manifest = json.loads(json.dumps(manifest, allow_nan=False))
    fingerprint = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    value = {**manifest, "signature": fingerprint}
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            old = json.loads(path.read_text())
            if old != value:
                raise RuntimeError("Output manifest differs; choose a new --output directory")
        else:
            atomic_json(path, value)


def jobs_for(args, config):
    rep_stop = config.repetitions if args.rep_stop is None else args.rep_stop
    if not 0 <= args.rep_start < rep_stop <= config.repetitions:
        raise ValueError("Require 0 <= rep-start < rep-stop <= repetitions")
    jobs = []
    for rep in range(args.rep_start, rep_stop):
        for n in config.n_values:
            for alpha in config.score_alphas:
                jobs.append((n, rep, alpha))
    return jobs


def payload_for(args, config, job):
    n, rep, alpha = job
    return (
        str(args.source.resolve()), [str(p.resolve()) for p in args.alpha_roots],
        str(args.output.resolve()), args.profile,
        tuple(config.n_values), tuple(config.score_alphas), config.repetitions,
        config.lmc_steps, config.particles, n, rep, alpha,
    )


def resolve_alpha_roots(args) -> list[Path]:
    roots = [path.resolve() for path in (args.alpha_output or [])]
    if args.alpha_root is not None:
        parent = args.alpha_root.resolve()
        roots.extend(sorted(parent.glob("batch_*/paper_results_2d")))
    unique = []
    seen = set()
    for root in roots:
        if root not in seen:
            unique.append(root)
            seen.add(root)
    if not unique:
        raise ValueError(
            "Provide --alpha-root containing batch_*/paper_results_2d, "
            "or one or more --alpha-output paths"
        )
    return unique


def summarize(output: Path) -> None:
    import pandas as pd
    from scipy.stats import t as student_t

    rows = []
    for path in sorted((output / "rows").glob("*.json")):
        rows.extend(json.loads(path.read_text()))
    if not rows:
        raise RuntimeError(f"No completed rows under {output / 'rows'}")
    frame = pd.DataFrame(rows).sort_values(["mode", "alpha_total", "n", "rep"])
    frame.to_csv(output / "raw.csv", index=False)
    metrics = [
        "coefficient_rms", "score_function_rms", "density_tv_change",
        "ou_tv_change", "generated_binned_tv_change",
        "generated_sliced_wasserstein_change", "paired_particle_rms",
    ]
    summaries = []
    for keys, group in frame.groupby(["mode", "n", "alpha_total"], sort=True):
        mode, n, alpha_total = keys
        for metric in metrics:
            values = group[metric].dropna().to_numpy(float)
            count = len(values)
            mean = float(np.mean(values))
            se = float(np.std(values, ddof=1) / np.sqrt(count)) if count > 1 else float("nan")
            critical = float(student_t.ppf(0.975, count - 1)) if count > 1 else float("nan")
            summaries.append({
                "mode": mode, "n": n, "alpha_total": alpha_total,
                "metric": metric, "count": count, "mean": mean, "se": se,
                "ci_low": mean - critical * se if count > 1 else float("nan"),
                "ci_high": mean + critical * se if count > 1 else float("nan"),
            })
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    print(f"Summarized {len(frame)} comparison rows from {len(frame) // 2} fits")


def progress(output: Path, expected: int | None) -> None:
    completed = len(list((output / "rows").glob("*.json")))
    failures = len(list((output / "failures").glob("*.json")))
    suffix = f"/{expected}" if expected is not None else ""
    print(f"completed={completed}{suffix}, failures={failures}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("run-job", "batch", "progress", "summarize"))
    parser.add_argument("--source", type=Path, default=Path(__file__).with_name("experiment.py"))
    parser.add_argument("--alpha-output", type=Path, nargs="+",
                        help="One or more completed paper_results_2d roots")
    parser.add_argument("--alpha-root", type=Path,
                        help="Parent containing batch_*/paper_results_2d")
    parser.add_argument("--output", type=Path, default=Path("fixed3_perturbation_results"))
    parser.add_argument("--profile", choices=("smoke", "pilot", "paper"), default="paper")
    parser.add_argument("--n-values", type=int, nargs="+")
    parser.add_argument("--score-alphas", type=float, nargs="+")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--lmc-steps", type=int)
    parser.add_argument("--particles", type=int)
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--rep-stop", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--job-id", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    args.source = args.source.resolve()
    args.alpha_roots = resolve_alpha_roots(args)
    args.output = args.output.resolve()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    engine = load_engine(args.source)
    config = config_from_args(engine, args)
    jobs = jobs_for(args, config)

    if args.task == "progress":
        progress(args.output, len(jobs))
        return
    if args.task == "summarize":
        summarize(args.output)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    ensure_manifest(
        args.output / "manifest.json",
        manifest_for(args.source, args.alpha_roots, config),
    )
    if args.task == "run-job":
        if args.job_id is None or not 0 <= args.job_id < len(jobs):
            raise ValueError(f"--job-id must lie in [0, {len(jobs) - 1}]")
        selected = [jobs[args.job_id]]
    else:
        selected = jobs

    payloads = [payload_for(args, config, job) for job in selected]
    failures = args.output / "failures"
    failures.mkdir(parents=True, exist_ok=True)
    try:
        if args.workers == 1 or len(payloads) == 1:
            for index, payload in enumerate(payloads, 1):
                run_one(payload)
                print(f"{index}/{len(payloads)} complete", flush=True)
        else:
            context = get_context("spawn")
            with context.Pool(args.workers) as pool:
                for index, _ in enumerate(pool.imap_unordered(run_one, payloads), 1):
                    print(f"{index}/{len(payloads)} complete", flush=True)
    except BaseException:
        failure = failures / f"failure_{os.getpid()}_{int(time.time())}.json"
        atomic_json(failure, {"traceback": traceback.format_exc(), "time": time.time()})
        raise


if __name__ == "__main__":
    main()
