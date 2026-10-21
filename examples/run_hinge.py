"""Load the hinge friction plugin and run ordinary MuJoCo steps without a viewer."""

import argparse
from pathlib import Path

import mujoco


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", type=Path, required=True,
                        help="Path to the compiled mjhinge shared library")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if not args.plugin.is_file():
        parser.error(f"Plugin library does not exist: {args.plugin}")
    if args.steps < 0:
        parser.error("--steps must be nonnegative")

    # Registration must happen before the XML is compiled.
    mujoco.mj_loadPluginLibrary(str(args.plugin.resolve()))
    model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name("hinge.xml")))
    data = mujoco.MjData(model)
    for _ in range(args.steps):
        mujoco.mj_step(model, data)

    print(f"MuJoCo {mujoco.__version__}: {model.nplugin} plugin instance(s)")
    print(f"time={data.time:.3f}, qpos={data.qpos}, qvel={data.qvel}")
    print(f"joint frictionloss={model.dof_frictionloss}")
    print(f"last radial/axial load={data.plugin_state}")
    print(f"constraint torque={data.qfrc_constraint}")


if __name__ == "__main__":
    main()

