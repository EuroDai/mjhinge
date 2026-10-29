"""Export a native C++ trajectory, then compare it on CPU/CUDA in another env.

Only export mode needs the C++ library. Comparison uses its saved history as
the oracle, so it can run on pc4 with the MuJoCo 3.16 Warp environment.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plugin", type=Path)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--qpos", type=float, nargs="+")
    parser.add_argument("--qvel", type=float, nargs="+")
    parser.add_argument("--ctrl", type=float, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--same-state", action="store_true",
                        help="Also compare accelerations at identical saved positions, velocities and load histories")
    args = parser.parse_args()
    if args.export:
        if not args.plugin or args.steps < 1:
            parser.error("export requires --plugin and positive --steps")
        mujoco.mj_loadPluginLibrary(str(args.plugin.resolve()))
        model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
        data = mujoco.MjData(model)
        for name in ("qpos", "qvel", "ctrl"):
            value = getattr(args, name)
            if value is not None:
                getattr(data, name)[:] = value
        initial = {name: getattr(data, name).copy() for name in ("qpos", "qvel", "ctrl")}
        records = {name: [] for name in ("qpos", "qvel", "qacc", "loads")}
        for _ in range(args.steps):
            mujoco.mj_step(model, data)
            for name in records:
                records[name].append((data.plugin_state if name == "loads" else getattr(data, name)).copy())
        args.export.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.export, **{k: np.asarray(v) for k, v in records.items()},
                            **{"initial_" + k: v for k, v in initial.items()},
                            mujoco_version=mujoco.__version__)
        print(f"Exported {args.steps} native steps to {args.export}")
        return
    if not args.reference:
        parser.error("choose --export or --reference")
    from mjhinge_warp import HingeSimulation
    reference = np.load(args.reference)
    sim = HingeSimulation(args.model, device=args.device)
    same = HingeSimulation(args.model, device=args.device) if args.same_state else None
    for name in ("qpos", "qvel", "ctrl"):
        getattr(sim.data, name).assign(reference["initial_" + name][None, :].astype(np.float32))
    errors = dict(qpos=0.0, qvel=0.0, qacc=0.0, loads=0.0)
    worst = {}
    same_state_error = 0.0
    for index in range(len(reference["qpos"])):
        sim.step()
        for name in errors:
            actual = sim.loads.numpy().reshape(-1) if name == "loads" else getattr(sim.data, name).numpy()[0]
            error = float(np.max(np.abs(actual - reference[name][index]), initial=0))
            if error > errors[name]:
                errors[name] = error
                worst[name] = dict(step=index + 1, actual=actual.tolist(),
                                   reference=reference[name][index].tolist())
        if same is not None:
            for name in ("qpos", "qvel"):
                value = reference["initial_" + name] if index == 0 else reference[name][index - 1]
                getattr(same.data, name).assign(value[None].astype(np.float32))
            same.set_control(reference["initial_ctrl"])
            history = np.zeros((1, same.nhinge, 2), dtype=np.float32) if index == 0 else reference["loads"][index - 1].reshape(1, same.nhinge, 2).astype(np.float32)
            same.loads.assign(history)
            same.forward()
            same_state_error = max(same_state_error, float(np.max(np.abs(
                same.data.qacc.numpy()[0] - reference["qacc"][index]))))
    sim.check()
    # Fixed float32 tolerances for saved reference trajectories.
    tolerances = dict(qpos=2e-4, qvel=2e-3, qacc=.2, loads=2e-3)
    passed = all(errors[k] <= tolerances[k] for k in errors)
    result = dict(passed=passed, device=str(sim.device), model=str(args.model),
                  steps=len(reference["qpos"]),
                  reference_mujoco=str(reference["mujoco_version"]),
                  warp_mujoco=mujoco.__version__, max_absolute_errors=errors,
                  worst_samples=worst,
                  tolerances=tolerances)
    if same is not None:
        result["same_state_max_qacc_error"] = same_state_error
        result["same_state_qacc_passed"] = same_state_error <= tolerances["qacc"]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
