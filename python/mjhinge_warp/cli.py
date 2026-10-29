"""Run a batch of mjhinge worlds without compiling the native plugin."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nworld", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--nconmax", type=int, default=64)
    parser.add_argument("--njmax", type=int, default=256)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 0 or args.warmup < 0:
        parser.error("steps and warmup must be nonnegative")
    from .simulation import HingeSimulation
    import warp as wp
    sim = HingeSimulation(args.model, nworld=args.nworld, device=args.device,
                          nconmax=args.nconmax, njmax=args.njmax)
    sim.step(args.warmup)
    sim.reset()
    if args.graph:
        sim.capture_step()
    wp.synchronize_device(sim.device)
    start = time.perf_counter()
    sim.step(args.steps)
    wp.synchronize_device(sim.device)
    elapsed = time.perf_counter() - start
    sim.check()
    result = dict(device=str(sim.device), nworld=sim.nworld, steps=args.steps,
                  hardware=sim.device.name,
                  versions={name: version(name) for name in ("mujoco", "mujoco-warp", "warp-lang")},
                  graph=args.graph, seconds=elapsed,
                  world_steps_per_second=args.nworld * args.steps / elapsed,
                  time=float(sim.data.time.numpy()[0]),
                  qpos_world0=sim.data.qpos.numpy()[0].tolist(),
                  loads_world0=sim.loads.numpy()[0].tolist(),
                  limits_world0=sim.limits.numpy()[0].tolist())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
