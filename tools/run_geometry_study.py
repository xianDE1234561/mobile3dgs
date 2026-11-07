"""Run the preregistered RGB-only geometry study, one GPU job at a time."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

import rgb_only_mcmc as core
from fuse_mvs import read_depth
from geometry_reference import depth_metrics, load_references
from mobile3dgs.initialization import initialize


PLAN = Path("experiments/geometry-accuracy-study.json")
SOURCE = Path("runs/analysis/rgbinit-mvs/initialization")
MVS = Path("runs/rgb-only/desk/mvs")
GEOMETRY_MANIFEST = Path("runs/analysis/rgbinit-mvs/prepared/frames.json")
MANIFEST = Path("runs/analysis/manifest-sharp.json")
FEATURES = Path("runs/rgb-only/pose-quickcheck/features.npz")


def dump(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    if json.loads(path.read_text()) != value:
        raise RuntimeError(f"Result did not persist: {path}")


def command(arguments, logfile):
    print(json.dumps({"running": arguments, "log": str(logfile)}), flush=True)
    with Path(logfile).open("w") as stream:
        subprocess.run([sys.executable, *arguments], stdout=stream, stderr=subprocess.STDOUT,
                       check=True, timeout=1800)


def quality(row):
    g = row["geometry"]
    mae = g["depth"]["mae_m"]
    cost = row.get("training", {}).get("training_seconds", 0.) + row.get("training", {}).get("setup_seconds", 0.)
    return (g["depth"]["within_5cm_fraction"], g["cloud"]["thresholds"]["0.02"]["fscore"],
            -mae if mae is not None else -float("inf"), -cost)


def passes_gates(candidate, reference):
    c, b = candidate["geometry"], reference["geometry"]
    if any(g["depth"]["mae_m"] is None or g["depth_bands"]["2-4m"]["mae_m"] is None for g in (c, b)):
        return False
    return (c["depth"]["coverage"] >= b["depth"]["coverage"]-.005
            and c["depth"]["within_2cm_fraction"] >= b["depth"]["within_2cm_fraction"]-.005
            and c["depth_bands"]["2-4m"]["mae_m"] <= b["depth_bands"]["2-4m"]["mae_m"]+.01)


def confirmed(candidate_rows, baseline_rows):
    if set(candidate_rows) != set(baseline_rows) or len(candidate_rows) < 3:
        return False
    if any(row["geometry"]["depth"]["mae_m"] is None
           or row["geometry"]["depth_bands"]["2-4m"]["mae_m"] is None
           for rows in (candidate_rows, baseline_rows) for row in rows.values()):
        return False
    differences = [quality(candidate_rows[s])[0]-quality(baseline_rows[s])[0] for s in baseline_rows]
    def mean_field(rows, *keys):
        values = []
        for row in rows.values():
            value = row["geometry"]
            for key in keys:
                value = value[key]
            values.append(value)
        return float(np.mean(values))
    return (np.mean(differences) > 0 and sum(d > 0 for d in differences) >= 2
            and mean_field(candidate_rows, "depth", "coverage") >= mean_field(baseline_rows, "depth", "coverage")-.005
            and mean_field(candidate_rows, "depth", "within_2cm_fraction") >= mean_field(baseline_rows, "depth", "within_2cm_fraction")-.005
            and mean_field(candidate_rows, "depth_bands", "2-4m", "mae_m") <= mean_field(baseline_rows, "depth_bands", "2-4m", "mae_m")+.01)


def point_depth(points, reference):
    c2w = reference["c2w"]
    camera = (points-c2w[:3, 3]) @ c2w[:3, :3]
    good = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0)
    camera = camera[good]
    fx, fy, cx, cy = reference["intrinsics"]
    pixels = camera[:, :2]/camera[:, 2:]*[fx, fy]+[cx, cy]
    height, width = reference["depth"].shape
    inside = ((pixels >= -.5) & (pixels < [width-.5, height-.5])).all(axis=1)
    pixels = np.floor(pixels[inside]+.5).astype(np.int64)
    z = camera[inside, 2]
    result = np.full(height*width, np.inf)
    np.minimum.at(result, pixels[:, 1]*width+pixels[:, 0], z)
    result = result.reshape(height, width)
    return result, np.isfinite(result).astype(np.float32)


def stage_diagnosis(root):
    destination = root/"stages"
    if (destination/"diagnosis.json").exists():
        return json.loads((destination/"diagnosis.json").read_text())
    destination.mkdir()
    manifest = json.loads(MANIFEST.read_text())
    geometry_manifest = json.loads(GEOMETRY_MANIFEST.read_text())
    references = load_references(manifest)
    fx, fy, cx, cy = geometry_manifest["intrinsics"]
    width, height = geometry_manifest["image_size"]
    v, u = np.mgrid[0:height:4, 0:width:4]
    pieces = []
    for index, frame in enumerate(geometry_manifest["frames"]):
        if frame["split"] != "train":
            continue
        depth = read_depth(MVS/"stereo/depth_maps"/f"{index:06d}.png.geometric.bin", width, height)[v, u]
        valid = np.isfinite(depth) & (depth > 0) & (depth <= 4)
        z = depth[valid]
        camera = np.column_stack(((u[valid]-cx)*z/fx, (v[valid]-cy)*z/fy, z))
        pose = np.asarray(frame["c2w"])
        pieces.append(camera @ pose[:3, :3].T+pose[:3, 3])
    raw = np.concatenate(pieces)
    with np.load(root/"initializations/candidates.npz", allow_pickle=False) as arrays:
        xyz, rgb = arrays["xyz"], arrays["rgb"]
    result = {"scope": "Fixed held-out reference; point z-buffers are sparse diagnostic proxies, not directly comparable to opacity-median Gaussian depths",
              "sensor_depth_used_for_construction": False, "point_stages": {}}
    for name, points in (("raw-mvs-stride4", raw), ("voxel-centroids", xyz)):
        predictions = [point_depth(points, r) for r in references]
        result["point_stages"][name] = {
            "count": len(points),
            "depth": depth_metrics(np.concatenate([r["depth"] for r in references]),
                                   np.concatenate([r["valid"] for r in references]),
                                   np.concatenate([p[0] for p in predictions]),
                                   np.concatenate([p[1] for p in predictions])),
            "frames": [{"index": r["index"], **depth_metrics(r["depth"], r["valid"], p[0], p[1])}
                       for r, p in zip(references, predictions)]}
    full = initialize(xyz, rgb, len(xyz), 42, .1)
    sources = {"retained": SOURCE,
               "dense-normal": root/"initializations/dense-normal/initialization",
               "depth-balanced": root/"initializations/depth-balanced/initialization"}
    checkpoints = {}
    for name in ("retained", "dense-normal", "depth-balanced", "full-cloud"):
        if name == "full-cloud":
            arrays = full
            params = {"means": torch.tensor(arrays["means"]), "scales": torch.tensor(arrays["scales"]).log(),
                      "quats": torch.nn.functional.normalize(torch.tensor(arrays["quats"]), dim=-1),
                      "opacities": torch.zeros(len(xyz)),
                      "sh0": ((torch.tensor(arrays["colors"])-.5)/.28209479177387814)[:, None, :]}
        else:
            values, _, _ = core.parameters_from_initialization(sources[name], device="cpu")
            params = {key: values[source].detach() for source, key in core.PARAMETERS.items()}
            params["opacities"].zero_()
        state = {"format": core.FORMAT, "manifest": manifest, "parameters": params,
                 "pixel_center_offset": .5, "step": 0, "config": {"initial_stage": name}}
        path = destination/f"{name}.pt"
        torch.save(state, path)
        checkpoints[f"init-{name}"] = path
    checkpoints["trained-featureba-history"] = Path("runs/rgb-only/pose-quickcheck/feature-ba-a/model.pt")
    arguments = ["tools/geometry_eval.py", "--manifest", str(MANIFEST), "--output", str(destination/"geometry")]
    for name, checkpoint in checkpoints.items():
        arguments.extend(["--model", f"{name}={checkpoint}"])
    command(arguments, destination/"geometry.log")
    result["gaussian_stages"] = json.loads((destination/"geometry/results.json").read_text())["models"]
    dump(destination/"diagnosis.json", result)
    return result


def preserve_incomplete(path, completion):
    """Keep interrupted artifacts, but allow the CLI to use a fresh destination."""
    path = Path(path)
    if path.exists() and not (path/completion).exists():
        preserved = path.with_name(f"{path.name}.incomplete-{time.time_ns()}")
        path.rename(preserved)
        return preserved
    return None


def run_one(root, name, candidate, seed, ledger):
    if any(core.fingerprint(p) != digest for p, digest in ledger.get("implementation_sha256", {}).items()):
        raise ValueError("Frozen experiment implementation changed")
    run_id = f"{name}-s{seed}"
    result_path = root/"measurements"/f"{run_id}.json"
    config = json.loads(Path("experiments/rgb-only-mcmc.json").read_text())
    config.update(seed=seed, mode="feature-ba", features=str(FEATURES),
                  mvs=str(MVS), geometry_manifest=str(GEOMETRY_MANIFEST),
                  init_opacity=candidate.get("init_opacity", .5), teacher_every=3,
                  depth_weight=candidate.get("depth_weight", 0.),
                  variance_weight=candidate.get("variance_weight", 0.))
    initialization = candidate.get("initialization", "retained")
    if initialization != "retained":
        config["initialization"] = str(root/"initializations"/initialization/"initialization")
    if "noise_lr" in candidate:
        config["noise_lr"] = candidate["noise_lr"]
    if result_path.exists():
        row = json.loads(result_path.read_text())
        if row["seed"] != seed or row["candidate"] != candidate or row["config"] != config:
            raise ValueError("Cached measurement belongs to a different experiment")
        if core.fingerprint(row["checkpoint"]) != row["checkpoint_sha256"]:
            raise ValueError("Cached checkpoint changed")
        ledger["runs"][run_id] = row
        return row
    config_path = root/"configs"/f"{run_id}.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing run configuration differs from requested experiment")
    if not config_path.exists():
        dump(config_path, config)
    training = root/"training"/run_id
    started = time.perf_counter()
    if not (training/"result.json").exists():
        preserve_incomplete(training, "result.json")
        command(["tools/train_geometry.py", "--config", str(config_path), "--output", str(training)], root/"logs"/f"{run_id}-train.log")
    train_result = json.loads((training/"result.json").read_text())
    if train_result["config"] != config:
        raise ValueError("Cached training configuration differs from requested experiment")
    saved_state = torch.load(training/"model.pt", map_location="cpu", weights_only=True)
    if saved_state["config"] != config:
        raise ValueError("Checkpoint configuration differs from requested experiment")
    del saved_state
    appearance = root/"appearance"/run_id
    if not (appearance/"results.json").exists():
        preserve_incomplete(appearance, "results.json")
        command(["tools/pose_align_eval.py", "--model", f"model={training/'model.pt'}", "--manifest", str(MANIFEST), "--output", str(appearance)], root/"logs"/f"{run_id}-appearance.log")
    geometry = root/"geometry"/run_id
    if not (geometry/"results.json").exists():
        preserve_incomplete(geometry, "results.json")
        command(["tools/geometry_eval.py", "--model", f"model={training/'model.pt'}", "--manifest", str(MANIFEST), "--output", str(geometry)], root/"logs"/f"{run_id}-geometry.log")
    row = {"name": name, "seed": seed, "candidate": candidate, "config": config,
           "config_path": str(config_path), "checkpoint": str(training/"model.pt"),
           "checkpoint_sha256": core.fingerprint(training/"model.pt"), "training": train_result,
           "appearance": json.loads((appearance/"results.json").read_text())["models"]["model"],
           "geometry": json.loads((geometry/"results.json").read_text())["models"]["model"],
           "end_to_end_measurement_seconds": time.perf_counter()-started}
    dump(result_path, row)
    ledger["runs"][run_id] = row
    dump(root/"results.json", ledger)
    print(json.dumps({"measured": run_id, "geometry": row["geometry"]["depth"],
                      "aligned": row["appearance"]["aligned"]}), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("diagnose", "screen", "confirm", "all"), default="all")
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    root = Path(plan["output"])
    root.mkdir(parents=True, exist_ok=True)
    for name in ("configs", "training", "appearance", "geometry", "measurements", "logs"):
        (root/name).mkdir(exist_ok=True)
    immutable_paths = [Path("experiments/rgb-only-mcmc.json"), SOURCE/"gaussians.npz", SOURCE/"experiment.json",
                       MANIFEST, GEOMETRY_MANIFEST, FEATURES, Path("tools/geometry_eval.py"),
                       Path("tools/geometry_reference.py"), Path("tools/gaussian_surface.py")]
    ledger_path = root/"results.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        if ledger["plan_sha256"] != core.fingerprint(PLAN):
            if ledger["runs"] or any((root/"training").iterdir()) or any((root/"measurements").iterdir()):
                raise ValueError("Cannot amend a plan after training artifacts exist")
            ledger.setdefault("pre_measurement_plan_amendments", []).append({
                "previous_plan": ledger["plan"], "previous_sha256": ledger["plan_sha256"],
                "reasons": plan.get("pre_measurement_amendments", []),
                "recorded_at": datetime.now(timezone.utc).isoformat()})
            ledger["plan"] = plan
            ledger["plan_sha256"] = core.fingerprint(PLAN)
            dump(ledger_path, ledger)
    else:
        ledger = {"started_at": datetime.now(timezone.utc).isoformat(), "plan": plan,
                  "plan_sha256": core.fingerprint(PLAN), "runs": {},
                  "immutable_sha256": {str(p): core.fingerprint(p) for p in immutable_paths}}
        dump(ledger_path, ledger)
    if any(core.fingerprint(p) != digest for p, digest in ledger["immutable_sha256"].items()):
        raise ValueError("Frozen input or evaluator changed")
    if not (root/"initializations/preparation.json").exists():
        command(["tools/study_initialization.py", "--output", str(root/"initializations")], root/"logs/initializations.log")
    if args.phase != "diagnose":
        implementation_paths = [
            Path("tools/train_geometry.py"), Path("tools/pose_align_eval.py"),
            Path(__file__), Path("tools/rgb_only_mcmc.py"),
            Path("tools/study_initialization.py"), Path("src/mobile3dgs/rendering.py"),
            Path("src/mobile3dgs/initialization.py"),
            root/"initializations/dense-normal/initialization/gaussians.npz",
            root/"initializations/depth-balanced/initialization/gaussians.npz"]
        if "implementation_sha256" not in ledger:
            if ledger["runs"] or any((root/"training").iterdir()):
                raise ValueError("Existing training has no frozen implementation record")
            ledger["implementation_sha256"] = {str(p): core.fingerprint(p) for p in implementation_paths}
            dump(ledger_path, ledger)
        if any(core.fingerprint(p) != digest for p, digest in ledger["implementation_sha256"].items()):
            raise ValueError("Training, appearance implementation or candidate initialization changed")
    if args.phase in ("diagnose", "all"):
        stage_diagnosis(root)
    if args.phase == "diagnose":
        return
    candidates = {c["name"]: c for c in plan["screen_candidates"]}
    if args.phase in ("screen", "all"):
        reference = run_one(root, "reference", candidates["reference"], 42, ledger)
        for name, candidate in candidates.items():
            if name != "reference":
                run_one(root, name, candidate, 42, ledger)
        beneficial = {name: ledger["runs"][f"{name}-s42"] for name in candidates if name != "reference"
                      and passes_gates(ledger["runs"][f"{name}-s42"], reference)
                      and quality(ledger["runs"][f"{name}-s42"]) > quality(reference)}
        init_names = [n for n in ("dense-normal", "depth-balanced") if n in beneficial]
        geometry_names = [n for n in ("mvs-depth-01", "mvs-depth-03", "mvs-depth-var") if n in beneficial]
        if geometry_names and (init_names or "no-noise" in beneficial or "low-opacity" in beneficial):
            best_geometry = max(geometry_names, key=lambda n: quality(beneficial[n]))
            combined = dict(candidates[best_geometry], name="combined")
            if init_names:
                best_init = max(init_names, key=lambda n: quality(beneficial[n]))
                combined["initialization"] = candidates[best_init]["initialization"]
            if "no-noise" in beneficial:
                combined["noise_lr"] = 0.
            if "low-opacity" in beneficial:
                combined["init_opacity"] = .1
            ledger["combined_candidate"] = combined
            dump(ledger_path, ledger)
            candidates["combined"] = combined
            run_one(root, "combined", combined, 42, ledger)
        eligible = [n for n in candidates if passes_gates(ledger["runs"][f"{n}-s42"], reference)]
        winner = max(eligible, key=lambda n: quality(ledger["runs"][f"{n}-s42"]))
        ledger["screen_winner"] = winner
        dump(ledger_path, ledger)
        print(json.dumps({"screen_winner": winner, "eligible": eligible}), flush=True)
    if args.phase in ("confirm", "all"):
        winner = ledger["screen_winner"]
        candidate = ledger["runs"][f"{winner}-s42"]["candidate"]
        for seed in plan["training"]["confirmation_seeds"]:
            run_one(root, "reference", candidates["reference"], seed, ledger)
            if winner != "reference":
                run_one(root, winner, candidate, seed, ledger)
        seeds = [42, *plan["training"]["confirmation_seeds"]]
        baseline_rows = {s: ledger["runs"][f"reference-s{s}"] for s in seeds}
        candidate_rows = {s: ledger["runs"][f"{winner}-s{s}"] for s in seeds}
        accepted = winner != "reference" and confirmed(candidate_rows, baseline_rows)
        chosen = winner if accepted else "reference"
        chosen_rows = candidate_rows if accepted else baseline_rows
        ledger["final"] = {"selected": chosen, "screen_winner": winner, "confirmation_passed": accepted,
                           "seeds": seeds,
                           "mean_within_5cm": {"reference": float(np.mean([quality(r)[0] for r in baseline_rows.values()])),
                                               "candidate": float(np.mean([quality(r)[0] for r in candidate_rows.values()]))},
                           "paired_primary_differences": {str(s): quality(candidate_rows[s])[0]-quality(baseline_rows[s])[0] for s in seeds},
                           "finished_at": datetime.now(timezone.utc).isoformat(),
                           "scope": plan["scope"], "mobile_measured": False}
        best_config = dict(chosen_rows[42]["config"])
        best_config["selection_evidence"] = str(ledger_path)
        best_config["rgb_evaluation"] = plan["rgb_evaluation"]
        best_config["geometry_evaluation"] = plan["geometry_evaluation"]
        dump(Path("experiments/rgb-only-geometry-best.json"), best_config)
        ledger["final"]["config"] = "experiments/rgb-only-geometry-best.json"
        if any(core.fingerprint(p) != digest for p, digest in ledger["immutable_sha256"].items()):
            raise ValueError("Frozen baseline changed during experiments")
        dump(ledger_path, ledger)
        print(json.dumps({"final": ledger["final"]}), flush=True)


if __name__ == "__main__":
    main()
