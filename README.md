# mjhinge

English | [简体中文](README_zh.md)

A standalone MuJoCo C++ plugin for load-dependent hinge friction. Load the shared library, configure the plugin in MJCF, and call `mj_step()` as usual. The simulation loop does not need to calculate bearing loads or update friction limits.

The plugin is registered as `mjhinge.hinge_friction`. Each instance targets a named hinge joint and supports load-dependent friction, optional low-speed static friction enhancement, and an angle-dependent radial gain. The joint's existing `frictionloss` and `damping` remain active. Optional terms default to zero, preserving existing configurations.

## MJCF configuration

```xml
<mujoco model="hinge_example">
  <extension>
    <plugin plugin="mjhinge.hinge_friction">
      <instance name="hinge_friction">
        <config key="joint" value="hinge"/>
        <config key="radial_gain" value="0.00045"/>
        <config key="axial_gain" value="0.00060"/>
      </instance>
    </plugin>
  </extension>

  <worldbody>
    <body name="link" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0"
             frictionloss="0.0015" damping="0"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0"
            size="0.025" mass="0.2"/>
    </body>
  </worldbody>
</mujoco>
```

| Parameter | Description | Units | Default |
| --- | --- | --- | --- |
| `joint` | Name of the target hinge joint | — | Required |
| `radial_gain` | Effective coefficient converting radial load to friction torque | N·m/N, or m | 0 |
| `axial_gain` | Effective coefficient converting axial load to friction torque | N·m/N, or m | 0 |
| `static_extra` | Additional low-speed friction limit | N·m | 0 |
| `stribeck_velocity` | Positive velocity scale for the additional friction decay | rad/s | 0.02 |
| `radial_peak_gain` | Amplitude of the Gaussian radial gain | m | 0 |
| `radial_peak_angle` | Joint angle at the radial gain peak | rad | 0.7 |
| `radial_peak_width` | Positive angular width of the radial gain peak | rad | 0.2 |
| `body_force_sensors` | Optional space-separated plugin sensor names; each outputs `3*nbody` world-frame force components | N | Empty |
| Joint `frictionloss` | MuJoCo's native joint friction torque limit | N·m | MuJoCo default |

The radial and axial gains must be finite and nonnegative, and already include the effective friction radius. Configure a separate instance for each joint; a joint may have only one instance. The target hinge must be the only joint connecting its body to its parent, so that the joint reaction has an unambiguous interpretation.

## Friction model

The friction torque limit at each step is:

```text
gain(q) = radial_gain + radial_peak_gain * exp(-0.5*((q-radial_peak_angle)/radial_peak_width)^2)
T_limit[n] = frictionloss + gain(q) * F_radial[n-1]
                         + axial_gain * F_axial[n-1]
                         + static_extra * exp(-(qvel/stribeck_velocity)^2)
```

Loads are derived from the joint interaction force `cfrc_int` computed by `mj_rnePostConstraint`, with a correction for spatial tendon routing forces omitted by MuJoCo 3.9. Let `a` be the joint's unit axis and `F` the interaction force, both in world coordinates:

```text
F_axial  = abs(dot(F, a))
F_radial = norm(F - dot(F, a) * a)
```

External cable plugins can report physical forces on each body through `body_force_sensors`. Each sensor must be a plugin sensor with dimension `3*nbody`, reporting world-frame `[Fx,Fy,Fz]` values in body-ID order. A sensor referenced by multiple hinge instances is counted only once. These forces are accumulated along body subtrees to complete the bearing load estimate. Sensors must remain enabled when this interface is configured. mjhinge does not link against a cable library and works independently when the interface is unused.

Finite-radius arc routes can use a geometry-only tendon with `gear="0"`, provided that its actuator's physical loads are reported through a configured body force sensor. Recovering loads from native arc tendons with nonzero gear is not supported. Native routes containing only sites and pulleys retain actuator-based load recovery, avoiding double counting for geometry-only transmissions.

The `advance` callback saves the most recently solved loads for use in the next step. Initialization and reset start with zero load history; the low-speed enhancement still uses the current velocity. Calling `mj_forward()` alone does not advance load history. With RK4, the stored reaction comes from the previous step's final dynamics stage.

The plugin updates friction constraint limits in the current `mjData`, including their island-local copies. MuJoCo's native constraint solver determines the magnitude and direction of static and sliding friction. No additional explicit friction torque is applied, and the shared `mjModel.dof_frictionloss` is not modified. Joint `solreffriction` and `solimpfriction` settings continue to control constraint softness, so small numerical slip can occur at rest.

Zero native `frictionloss` is supported. When the load-dependent limit becomes positive, the plugin rebuilds the current data's constraints through a local model view to insert the joint's friction row. This requires one additional constraint construction. For positive native `frictionloss`, the plugin updates the existing row directly.

Each instance stores its `[F_radial, F_axial]` history in `plugin_state`, supporting data copies, resets, and state serialization. Multiple `mjData` instances can share a model while retaining independent load histories. The plugin does not install global physics callbacks.

This is an effective model based on the net joint reaction. It does not reconstruct interference-fit pressure distributions, end-face separation, local compression caused by bearing moments, or load sharing between two supports. Reaction estimates follow the `cfrc_int` semantics of the MuJoCo version in use. External modules that provide only generalized torques cannot uniquely determine physical radial and axial loads. Calibrate the two gains against measurements, and check timestep sensitivity under rapid load changes and impacts.

## Build and run

The C++ implementation is compiled and validated against **MuJoCo 3.9.0** and uses that version's island data layout. Building requires CMake 3.20+, a C++17 compiler, and a matching MuJoCo SDK. Python examples and tests additionally require `mujoco` and `numpy`; Python is not a runtime dependency of the native plugin.

Windows PowerShell:

```powershell
cmake -S . -B build -A x64 -DCMAKE_TRY_COMPILE_CONFIGURATION=Release -DMUJOCO_ROOT="D:/path/to/mujoco"
cmake --build build --config Release
python examples/run_hinge.py --plugin build/Release/mjhinge.dll
ctest --test-dir build -C Release --output-on-failure
```

Alternatively, point `-Dmujoco_DIR=...` to the SDK's CMake configuration directory, or set `MUJOCO_INCLUDE_DIR` and `MUJOCO_LIBRARY` separately. The Windows SDK must include headers and the `mujoco.lib` import library. On Linux/macOS, omit `-A x64` and use `mjhinge.so` / `mjhinge.dylib`, respectively. The host loading the plugin must be able to find the matching MuJoCo runtime library.

In Python, register the shared library before compiling the MJCF, then step normally:

```python
from pathlib import Path
import mujoco

mujoco.mj_loadPluginLibrary(str(Path("build/Release/mjhinge.dll").resolve()))
model = mujoco.MjModel.from_xml_path("examples/hinge.xml")
data = mujoco.MjData(model)

for _ in range(1000):
    mujoco.mj_step(model, data)
```

C/C++ hosts can likewise register the shared library before loading the model and then use the standard simulation API. See the [MuJoCo extension documentation](https://mujoco.readthedocs.io/en/stable/programming/extension.html) for the plugin interface.

## Validation

```powershell
python tests/test_hinge_friction.py --plugin build-x64/Release/mjhinge.dll -v
python examples/run_hinge.py --plugin build-x64/Release/mjhinge.dll
```

Tests cover independent radial and axial responses, the one-step delay and unloading, friction direction, static holding and breakaway, native behavior with zero gains, zero `frictionloss`, world-frame axes, gravity and subtree loads, velocity sampling before integration, multiple instances, copy/reset/state replay, concurrent data sharing one model, split stepping, disable flags, and configuration validation.

For scenes with contacts and joint equality constraints, complete trajectories are compared with a reference model whose native `frictionloss` is updated externally. Coverage includes Newton / CG / PGS solvers and dense / sparse Jacobians. Basic execution is tested with Euler, implicit, implicitfast, and RK4 integrators.

The local Windows workspace uses headers and a DLL from the Python wheel, plus an import library generated in an ignored build directory. Its plugin is located at `build-x64/Release/mjhinge.dll`. For a fresh checkout, configure the SDK as described above.

## MJWarp / CUDA backend

The independent `mjhinge_warp` Python package reads the same plugin MJCF and converts it into a MuJoCo Warp model without requiring a C++ shared library. Each world's radial/axial load history stays on the device, and friction is handled by the native constraint solver. The backend supports the existing gains, angle-dependent peaks, low-speed friction enhancement, zero `frictionloss`, the one-step load delay, and independent world resets. `forward()` does not advance history.

The Warp backend is pinned to **MuJoCo / mujoco-warp 3.16.0 and Warp 1.18.0**. Install it in a separate environment from the C++ / MuJoCo 3.9 backend. On Linux:

```bash
python3 -m venv .venv-warp
source .venv-warp/bin/activate
python -m pip install -r requirements-warp.txt
python -m pip install -e . --no-deps
mjhinge-warp examples/hinge.xml --device cuda:0 --nworld 2048 --steps 1000 --graph
```

For Linux x86_64 with CUDA 12 drivers, including pc4's R550 driver, use `requirements-warp-cu12.txt` instead. On Windows, activate the environment with `.venv-warp\Scripts\Activate.ps1`. For CPU validation, select `--device cpu` and omit `--graph`. The [MJWarp guide (Chinese)](docs/MJWARP_BACKEND.md) documents the API, supported features, limitations, and cross-backend validation commands.

```python
from mjhinge_warp import HingeSimulation

sim = HingeSimulation("examples/hinge.xml", nworld=256, device="cuda:0")
sim.step(5)                 # Warm up kernels before first use.
sim.reset()
sim.capture_step()          # Optional: capture physics and history updates.
sim.step(1000)
sim.reset([0, 7])           # Reset physics and load history for selected worlds.
sim.check()                # Synchronize and check numerical state/capacities.
print(sim.loads.numpy())   # Shape: (nworld, nhinge, 2), [F_radial, F_axial].
```

`data.qpos`, `data.qvel`, `data.ctrl`, and `data.xfrc_applied` are device arrays; update their contents in place during training. MuJoCo 3.16's RNE already includes native spatial tendon forces, so the Warp backend does not repeat the C++ 3.9 correction. External C++ cable plugins, `body_force_sensors`, and sleeping bodies are not yet supported and are explicitly rejected during loading. Use native tendons or physical `xfrc_applied` forces instead. Advance through `sim.step()`; calling `mjw.step()` directly skips the independent history update.

## Independent analytical benchmarks

Run `python tests/validate_theory.py --plugin build-x64/Release/mjhinge.dll` to reproduce the analytical benchmarks and write JSON results and comparison plots to `validation/`. Plotting additionally requires matplotlib. Expected values are calculated independently from geometry, external loads, and rigid-body equations, without reading the plugin's internal friction limits.

The benchmark uses a solid sphere of mass 1 kg and radius 0.1 m rotating about its center, with inertia `I = 2 m r² / 5 = 0.004 kg·m²`. Gravity and viscous damping are disabled. Radial and axial forces act at the fixed center and produce no driving torque. With `frictionloss=0.002`, `radial_gain=0.003`, `axial_gain=0.004`, and loads of 3 N and 4 N, the sliding friction torque is `T = 0.027 N·m`.

After one stationary step initializes the load history, the sphere is given an initial angular velocity of `ω₀ = 5 rad/s`. Before stopping:

```text
α = -T/I
ω(t) = ω₀ - Tt/I
θ(t) = ω₀t - Tt²/(2I)
t_stop = Iω₀/T
θ_stop = Iω₀²/(2T)
```

Results with MuJoCo 3.9.0, a 1 ms Euler timestep, `solreffriction="0.002 1"`, and `solimpfriction="0.9999 0.9999 0.001"`:

| Quantity | Theory | Simulation |
| --- | --- | --- |
| Sliding friction torque | 0.027 N·m | 0.027 N·m |
| Angular acceleration | −6.75 rad/s² | −6.75 rad/s² |
| Angular velocity at 0.4 s | 2.3 rad/s | 2.30000000000001 rad/s |
| Rotation angle at 0.4 s | 1.46 rad | 1.45865 rad |
| Stopping time | 0.740741 s | 0.741 s (first speed below 10⁻⁶ rad/s) |
| Stopping angle | 1.851852 rad | 1.8493525 rad |

The angle error before stopping matches the analytical semi-implicit Euler discretization error, `θ_h − θ = α t h / 2`. As the timestep decreases from 2 to 1, 0.5, and 0.25 ms, the absolute angle error at 0.4 s decreases from 0.0027 to 0.00135, 0.000675, and 0.0003375 rad, demonstrating first-order convergence.

Additional independent checks cover 16 load combinations, inferring friction torque from angular acceleration with a maximum absolute error of 1.74×10⁻¹⁸ N·m; symmetry between rotation directions; energy dissipation; driving torque thresholds; and the one-step delay after load changes. An offset sphere is also evaluated over 500 steps using independent recurrences based on `I = I_cm + mL²`, `F_radial = mL sqrt(ω⁴ + α²)`, and `F_axial = mg`, checking the contributions of centripetal and tangential acceleration to the reaction. The maximum reaction error is 6.67×10⁻¹⁶ N.

**Default constraint parameters do not imply perfectly static behavior.** Under a driving torque of 0.02 N·m, below the 0.027 N·m friction limit, default parameters produce a steady slip velocity of approximately 0.005278 rad/s. The stiffer parameters above reduce it to approximately 5×10⁻⁷ rad/s. At a driving torque of 0.05 N·m, both parameter sets produce `(0.05 − 0.027)/0.004 = 5.75 rad/s²`. These benchmarks do not modify the example's solver settings or the plugin algorithm.

![Analytical predictions and simulation results](validation/theory_validation.png)
