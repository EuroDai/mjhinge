"""Independent analytical benchmarks for mjhinge (SI units throughout)."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


MASS, RADIUS = 1.0, 0.1
INERTIA = 2 * MASS * RADIUS**2 / 5
BASE, KR, KA = 0.002, 0.003, 0.004
FR, FA = 3.0, 4.0
CAP = BASE + KR * FR + KA * FA


def model(dt=0.001, hard=True, offset=0.0, gravity=0.0):
    # An explicit sphere geometry gives I = 2 m r^2 / 5 + m L^2.
    friction_solver = ('solreffriction="0.002 1" '
                       'solimpfriction="0.9999 0.9999 0.001"') if hard else ''
    xml = f'''<mujoco>
      <option timestep="{dt}" gravity="0 0 {-gravity}" integrator="Euler"
              iterations="100" tolerance="1e-12"/>
      <extension><plugin plugin="mjhinge.hinge_friction"><instance name="f">
        <config key="joint" value="hinge"/>
        <config key="radial_gain" value="{KR}"/>
        <config key="axial_gain" value="{KA}"/>
      </instance></plugin></extension>
      <worldbody><body>
        <joint name="hinge" axis="0 0 1" frictionloss="{BASE}"
               damping="0" {friction_solver}/>
        <geom type="sphere" size="{RADIUS}" mass="{MASS}"
              pos="{offset} 0 0" contype="0" conaffinity="0"/>
      </body></worldbody>
    </mujoco>'''
    m = mujoco.MjModel.from_xml_string(xml)
    return m, mujoco.MjData(m)


def run_spin(dt=0.001, hard=True, drive=0.0, omega=5.0, duration=1.0):
    m, d = model(dt=dt, hard=hard)
    # Force acts at the fixed centre of mass and therefore produces no torque.
    d.xfrc_applied[1, :3] = [FR, 0, FA]
    mujoco.mj_step(m, d)  # collect the load with the rotor at rest
    d.qvel[0] = omega
    d.qfrc_applied[0] = drive
    samples = [[0.0, 0.0, omega, 0.0, 0.0]]
    for n in range(round(duration / dt)):
        mujoco.mj_step(m, d)
        samples.append([(n + 1) * dt, d.qpos[0], d.qvel[0],
                        d.qacc[0], d.qfrc_constraint[0]])
    return np.asarray(samples)


def validate():
    results = {"mujoco_version": mujoco.__version__, "parameters": {
        "mass_kg": MASS, "sphere_radius_m": RADIUS,
        "inertia_kg_m2": INERTIA, "frictionloss_Nm": BASE,
        "radial_gain_m": KR, "axial_gain_m": KA,
        "radial_load_N": FR, "axial_load_N": FA,
        "theoretical_friction_Nm": CAP,
    }}

    # Measure friction from acceleration, independently of efc/plugin state.
    sweep = []
    for radial in (0.0, 1.0, 3.0, 10.0):
        for axial in (0.0, 2.0, 4.0, 10.0):
            m, d = model()
            d.xfrc_applied[1, :3] = (radial, 0, axial)
            mujoco.mj_step(m, d)
            d.qvel[0] = 5
            mujoco.mj_step(m, d)
            theory = BASE + KR * radial + KA * axial
            measured = -INERTIA * d.qacc[0]
            sweep.append([radial, axial, theory, float(measured)])
    sweep = np.asarray(sweep)
    error = float(np.max(np.abs(sweep[:, 2] - sweep[:, 3])))
    assert error < 1e-12, error
    results["load_sweep"] = {"cases": len(sweep), "max_torque_error_Nm": error,
                             "columns": ["Fr_N", "Fa_N", "theory_Nm", "from_acceleration_Nm"],
                             "values": sweep.tolist()}

    alpha = -CAP / INERTIA
    stop_time = 5.0 / -alpha
    stop_angle = INERTIA * 5.0**2 / (2 * CAP)
    spin = run_spin()
    row = spin[400]
    results["free_deceleration"] = {
        "theory_alpha_rad_s2": alpha, "measured_alpha_rad_s2": float(row[3]),
        "theory_omega_at_0_4s": 5 + alpha * 0.4,
        "measured_omega_at_0_4s": float(row[2]),
        "theory_angle_at_0_4s": 5 * 0.4 + alpha * 0.4**2 / 2,
        "measured_angle_at_0_4s": float(row[1]),
        "theory_stop_time_s": stop_time,
        "measured_time_below_1e_6_rad_s": float(spin[np.flatnonzero(abs(spin[:, 2]) < 1e-6)[0], 0]),
        "theory_stop_angle_rad": stop_angle,
        "measured_final_angle_rad": float(spin[-1, 1]),
    }
    assert abs(row[3] - alpha) < 1e-10
    assert abs(row[2] - (5 + alpha * 0.4)) < 1e-10
    assert abs(spin[-1, 1] - stop_angle) < 0.003
    # Friction must dissipate energy; midpoint work is exact for each constant
    # acceleration Euler velocity update, even in the softened stopping region.
    energy = 0.5 * INERTIA * spin[:, 2]**2
    work = np.sum(spin[1:, 4] * (spin[:-1, 2] + spin[1:, 2]) / 2 * 0.001)
    assert np.max(np.diff(energy)) < 1e-14
    assert abs((energy[-1] - energy[0]) - work) < 1e-12
    negative = run_spin(omega=-5)
    assert np.max(abs(negative[:, 1:3] + spin[:, 1:3])) < 1e-12
    results["energy"] = {"initial_J": float(energy[0]), "final_J": float(energy[-1]),
                         "friction_work_J": float(work),
                         "balance_error_J": float(abs((energy[-1] - energy[0]) - work)),
                         "positive_negative_motion_symmetric": True}

    convergence = []
    for dt in (0.002, 0.001, 0.0005, 0.00025):
        trace = run_spin(dt=dt)
        row = trace[round(0.4 / dt)]
        # Semi-implicit Euler's expected position error before stopping:
        # theta_h - theta_exact = alpha * t * h / 2.
        expected_error = alpha * 0.4 * dt / 2
        measured_error = row[1] - (5 * 0.4 + alpha * 0.4**2 / 2)
        assert abs(measured_error - expected_error) < 1e-11
        convergence.append({"dt_s": dt, "angle_error_at_0_4s_rad": float(measured_error),
                            "theory_euler_error_rad": expected_error,
                            "final_angle_error_rad": float(trace[-1, 1] - stop_angle)})
    results["step_size_convergence"] = convergence

    thresholds = []
    for hard in (False, True):
        for drive in (-0.05, -0.026, -0.02, 0.0, 0.02, 0.026, 0.05):
            trace = run_spin(hard=hard, drive=drive, omega=0.0)
            expected_accel = np.sign(drive) * max(abs(drive) - CAP, 0) / INERTIA
            assert abs(trace[-1, 3] - expected_accel) < 1e-9
            if abs(drive) < CAP and hard:
                assert abs(trace[-1, 2]) < 1e-5
            thresholds.append({"constraint": "stiff" if hard else "default",
                               "drive_Nm": drive, "ideal_alpha_rad_s2": float(expected_accel),
                               "measured_alpha_rad_s2": float(trace[-1, 3]),
                               "omega_at_1s_rad_s": float(trace[-1, 2])})
    results["drive_threshold"] = thresholds

    # Off-axis body: support load must include centripetal AND tangential
    # acceleration. Derive both directly from rigid-body dynamics, without RNE.
    length, gravity = 0.3, 9.81
    inertia = INERTIA + MASS * length**2
    m, d = model(offset=length, gravity=gravity)
    d.qvel[0] = 2
    expected_omega, previous_radial, previous_axial = 2.0, 0.0, 0.0
    reaction_error, velocity_error = 0.0, 0.0
    for _ in range(500):
        torque = BASE + KR * previous_radial + KA * previous_axial
        angular_accel = -torque / inertia
        radial = MASS * length * np.hypot(expected_omega**2, angular_accel)
        axial = MASS * gravity
        expected_omega += angular_accel * 0.001
        mujoco.mj_step(m, d)
        reaction_error = max(reaction_error, abs(d.plugin_state[0] - radial),
                             abs(d.plugin_state[1] - axial))
        velocity_error = max(velocity_error, abs(d.qvel[0] - expected_omega))
        previous_radial, previous_axial = radial, axial
    assert reaction_error < 1e-10, reaction_error
    assert velocity_error < 1e-10, velocity_error
    results["off_axis_dynamics"] = {"offset_m": length, "inertia_kg_m2": inertia,
        "max_reaction_error_N": float(reaction_error),
        "max_omega_error_rad_s": float(velocity_error),
        "final_omega_rad_s": float(d.qvel[0])}

    m, d = model()
    d.qvel[0] = 5
    applied_radial = [0, 3, 3, 10, 10, 0, 0]
    measured, predicted = [], []
    previous_load = 0
    for radial in applied_radial:
        d.xfrc_applied[1, 0] = radial
        mujoco.mj_step(m, d)
        measured.append(float(-INERTIA * d.qacc[0]))
        predicted.append(BASE + KR * previous_load)
        previous_load = radial
    np.testing.assert_allclose(measured, predicted, atol=1e-12, rtol=0)
    results["load_step_delay"] = {"radial_load_N": applied_radial,
                                  "theory_torque_Nm": predicted,
                                  "measured_torque_Nm": measured}
    results["passed"] = True
    return results, spin


def plot(results, spin, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    t = spin[:, 0]
    alpha = CAP / INERTIA
    stop = 5 / alpha
    axes[0, 0].plot(t, np.maximum(5 - alpha * t, 0), label="Ideal Coulomb", lw=2)
    axes[0, 0].plot(t, spin[:, 2], "--", label="Plugin, dt=1 ms")
    axes[0, 0].set(xlabel="Time (s)", ylabel="Angular velocity (rad/s)", title="Free deceleration")
    theta = 5 * np.minimum(t, stop) - alpha * np.minimum(t, stop)**2 / 2
    axes[0, 1].plot(t, theta, label="Analytical", lw=2)
    axes[0, 1].plot(t, spin[:, 1], "--", label="Semi-implicit Euler")
    axes[0, 1].set(xlabel="Time (s)", ylabel="Angle (rad)", title="Stopping angle")
    sweep = np.asarray(results["load_sweep"]["values"])
    axes[1, 0].plot([0, 0.075], [0, 0.075], label="Ideal agreement", lw=1)
    axes[1, 0].scatter(sweep[:, 2], sweep[:, 3], label="16 load combinations", s=22)
    axes[1, 0].set(xlabel="Theoretical torque (N m)", ylabel="Torque from acceleration (N m)",
                   title="Load dependence")
    conv = results["step_size_convergence"]
    axes[1, 1].plot([x["dt_s"] * 1000 for x in conv],
                    [abs(x["angle_error_at_0_4s_rad"]) for x in conv], "o-", label="Measured")
    axes[1, 1].set(xlabel="Time step (ms)", ylabel="Angle error at 0.4 s (rad)",
                   title="First-order integration error")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("validation"))
    args = parser.parse_args()
    mujoco.mj_loadPluginLibrary(str(args.plugin.resolve()))
    results, spin = validate()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "theory_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plot(results, spin, args.output / "theory_validation.png")
    print(json.dumps({key: value for key, value in results.items() if key != "load_sweep"}, indent=2))
