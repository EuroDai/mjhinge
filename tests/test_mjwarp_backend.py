"""Physical and lifecycle checks; MJHINGE_TEST_DEVICE=cpu or cuda:0."""

import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import warp as wp

from mjhinge_warp import HingeSimulation, UnsupportedHingeError, load_model, load_xml

DEVICE = os.environ.get("MJHINGE_TEST_DEVICE", "cpu")
ROOT = Path(__file__).resolve().parents[1]


def xml_model(base=0.002, radial=0.003, axial=0.004, integrator="Euler",
              extra="", axis="0 1 0", quat="1 0 0 0", geom_pos="0 0 0"):
    return f"""<mujoco>
      <option gravity="0 0 0" timestep="0.001" integrator="{integrator}"
              iterations="100" tolerance="1e-10"/>
      <extension><plugin plugin="mjhinge.hinge_friction"><instance name="f">
        <config key="joint" value="hinge"/>
        <config key="radial_gain" value="{radial}"/>
        <config key="axial_gain" value="{axial}"/>{extra}
      </instance></plugin></extension>
      <worldbody><body name="link" quat="{quat}">
        <joint name="hinge" axis="{axis}" frictionloss="{base}"
               solreffriction="0.002 1" solimpfriction="0.9999 0.9999 0.001"/>
        <geom type="sphere" pos="{geom_pos}" size="0.1" mass="1"
              contype="0" conaffinity="0"/>
      </body></worldbody>
    </mujoco>"""


def simulation(xml=None, nworld=1, **kwargs):
    return HingeSimulation(load_xml(xml or xml_model()), device=DEVICE,
                           nworld=nworld, **kwargs)


def set_force(sim, forces):
    values = np.zeros((sim.nworld, sim.cpu_model.nbody, 6), dtype=np.float32)
    values[:, 1, :3] = forces
    sim.data.xfrc_applied.assign(values)


def speed(sim, values):
    sim.data.qvel.assign(np.broadcast_to(np.asarray(values, dtype=np.float32),
                                         (sim.nworld, sim.cpu_model.nv)).copy())


def row_limit(sim, world=0, dof=0):
    types = sim.data.efc.type.numpy()[world]
    ids = sim.data.efc.id.numpy()[world]
    count = sim.data.nefc.numpy()[world]
    rows = np.flatnonzero((types[:count] == 1) & (ids[:count] == dof))
    return sim.data.efc.frictionloss.numpy()[world, rows[0]] if rows.size else 0.0


@pytest.mark.parametrize("base", [0, 0.002])
@pytest.mark.parametrize("integrator", ["Euler", "implicitfast", "implicit", "RK4"])
def test_delay_unload_and_integrators(base, integrator):
    sim = simulation(xml_model(base=base, integrator=integrator), nworld=3)
    speed(sim, [[5], [-5], [5]])
    set_force(sim, [[3, 4, 0], [0, -4, 0], [3, 0, 4]])
    sim.forward()
    sim.forward()
    np.testing.assert_array_equal(sim.loads.numpy(), 0)
    sim.step()
    np.testing.assert_allclose(sim.loads.numpy()[:, 0], [[3, 4], [0, 4], [5, 0]], atol=2e-5)
    assert row_limit(sim) == pytest.approx(base, abs=1e-8)
    sim.step()
    np.testing.assert_allclose(sim.limits.numpy()[:, 0], np.array([.025, .016, .015]) + base, atol=1e-7)
    np.testing.assert_allclose(sim.data.qfrc_constraint.numpy()[:, 0],
                               [-base-.025, base+.016, -base-.015], atol=2e-6)
    set_force(sim, [0, 0, 0])
    sim.step()
    assert row_limit(sim) == pytest.approx(base + .025, abs=1e-7)
    sim.step()
    assert row_limit(sim) == pytest.approx(base, abs=1e-7)
    np.testing.assert_array_equal(sim.cpu_model.dof_frictionloss, [base])
    sim.check()


def test_static_hold_breakaway_and_dissipation():
    sim = simulation(xml_model(base=0), nworld=2)
    set_force(sim, [3, 4, 0])
    sim.step()
    sim.data.qfrc_applied.assign(np.array([[.01], [.05]], dtype=np.float32))
    sim.step(100)
    assert abs(sim.data.qvel.numpy()[0, 0]) < 1e-5
    assert sim.data.qvel.numpy()[1, 0] > .1
    np.testing.assert_allclose(sim.data.qfrc_constraint.numpy()[:, 0], [-.01, -.025], atol=1e-6)
    assert np.all(sim.data.qfrc_constraint.numpy() * sim.data.qvel.numpy() <= 1e-10)


def test_angle_gain_and_stribeck():
    extra = """<config key="static_extra" value=".01"/>
    <config key="stribeck_velocity" value=".5"/>
    <config key="radial_peak_gain" value=".003"/>
    <config key="radial_peak_angle" value=".7"/>
    <config key="radial_peak_width" value=".2"/>"""
    sim = simulation(xml_model(base=0, radial=0, axial=0, extra=extra), nworld=3)
    q = np.array([[0], [.7], [1.4]], dtype=np.float32)
    v = np.array([[0], [.5], [2]], dtype=np.float32)
    sim.data.qpos.assign(q)
    sim.data.qvel.assign(v)
    sim.loads.assign(np.array([[[2, 0]]] * 3, dtype=np.float32))
    sim.forward()
    expected = .006 * np.exp(-.5 * ((q[:, 0]-.7)/.2)**2) + .01 * np.exp(-(v[:, 0]/.5)**2)
    np.testing.assert_allclose(sim.limits.numpy()[:, 0], expected, rtol=3e-6, atol=1e-8)
    np.testing.assert_array_equal(sim.loads.numpy()[:, 0], [[2, 0]] * 3)


def test_reset_independence_and_forward_history():
    sim = simulation(nworld=3)
    set_force(sim, [[3, 4, 0], [6, 8, 0], [9, 12, 0]])
    speed(sim, 2)
    sim.step(4)
    history = sim.loads.numpy().copy()
    qpos = sim.data.qpos.numpy().copy()
    times = sim.data.time.numpy().copy()
    sim.forward()
    np.testing.assert_array_equal(sim.loads.numpy(), history)
    sim.reset([1])
    np.testing.assert_array_equal(sim.loads.numpy()[[0, 2]], history[[0, 2]])
    np.testing.assert_array_equal(sim.data.qpos.numpy()[[0, 2]], qpos[[0, 2]])
    np.testing.assert_array_equal(sim.data.time.numpy()[[0, 2]], times[[0, 2]])
    np.testing.assert_array_equal(sim.loads.numpy()[1], 0)
    assert sim.data.time.numpy()[1] == 0
    assert row_limit(sim, 1) == pytest.approx(.002)
    sim.reset()
    np.testing.assert_array_equal(sim.loads.numpy(), 0)
    np.testing.assert_array_equal(sim.data.qvel.numpy(), 0)


@pytest.mark.parametrize("flag", ["mjDSBL_CONSTRAINT", "mjDSBL_FRICTIONLOSS"])
def test_disable_flags(flag):
    sim = simulation()
    set_force(sim, [3, 4, 0])
    sim.step(2)
    old = sim.loads.numpy().copy()
    sim.model.opt.disableflags = int(getattr(mujoco.mjtDisableBit, flag))
    set_force(sim, [6, 8, 0])
    sim.step()
    np.testing.assert_allclose(sim.data.qfrc_constraint.numpy(), 0, atol=1e-9)
    np.testing.assert_array_equal(sim.loads.numpy(), old)
    sim.model.opt.disableflags = 0
    sim.step()
    assert row_limit(sim) == pytest.approx(.027, abs=1e-7)


def test_world_axis_gravity_and_preintegration_velocity():
    sim = simulation(xml_model(axis="1 0 0", quat="1 0 0 1"))
    set_force(sim, [3, 4, 0])
    sim.step()
    np.testing.assert_allclose(sim.loads.numpy()[0, 0], [3, 4], atol=2e-5)
    # Offset sphere: centrifugal and tangential acceleration contribute to load.
    sim = simulation(xml_model(axis="0 0 1", geom_pos=".3 0 0"))
    speed(sim, 2)
    cpu = mujoco.MjData(sim.cpu_model)
    cpu.qvel[0] = 2
    mujoco.mj_forward(sim.cpu_model, cpu)
    mujoco.mj_rnePostConstraint(sim.cpu_model, cpu)
    expected = np.linalg.norm(cpu.cfrc_int[1, 3:5])
    sim.step()
    assert sim.loads.numpy()[0, 0, 0] == pytest.approx(expected, rel=2e-5)
    root = ET.fromstring(xml_model())
    root.find("option").set("gravity", "0 0 -10")
    child = ET.SubElement(root.find("worldbody/body"), "body")
    ET.SubElement(child, "geom", type="sphere", size=".05", mass=".5", contype="0", conaffinity="0")
    sim = simulation(ET.tostring(root, encoding="unicode"))
    sim.step()
    np.testing.assert_allclose(sim.loads.numpy()[0, 0], [15, 0], atol=1e-5)


def tendon_model():
    root = ET.fromstring(xml_model(base=0))
    world = root.find("worldbody")
    ET.SubElement(world, "site", name="a", pos=".3 .4 0")
    ET.SubElement(world, "site", name="b", pos="-.3 .4 0")
    ET.SubElement(world.find("body"), "site", name="centre", pos="0 0 0")
    spatial = ET.SubElement(ET.SubElement(root, "tendon"), "spatial", name="cable")
    for tag, attrs in [("site", dict(site="a")), ("site", dict(site="centre")),
                       ("pulley", dict(divisor="2")), ("site", dict(site="b")),
                       ("site", dict(site="centre"))]:
        ET.SubElement(spatial, tag, **attrs)
    ET.SubElement(ET.SubElement(root, "actuator"), "motor", tendon="cable", gear="3")
    return root


@pytest.mark.parametrize("clamp", [False, True])
def test_native_tendon_load_not_double_counted(clamp):
    root = tendon_model()
    if clamp:
        root.find("tendon/spatial").set("actuatorfrclimited", "true")
        root.find("tendon/spatial").set("actuatorfrcrange", "-1 1")
    sim = simulation(ET.tostring(root, encoding="unicode"))
    sim.set_control([2])
    sim.step(2)
    expected = np.array([1.8, 7.2]) * (.5 if clamp else 1)
    np.testing.assert_allclose(sim.loads.numpy()[0, 0], expected, rtol=2e-5, atol=1e-5)
    assert row_limit(sim) == pytest.approx(expected @ [.003, .004], rel=2e-5)


def test_multiple_instances_and_zero_gain_native_equivalence():
    root = ET.fromstring(xml_model())
    plugin = root.find("extension/plugin")
    second = ET.SubElement(plugin, "instance", name="second")
    ET.SubElement(second, "config", key="joint", value="other")
    ET.SubElement(second, "config", key="radial_gain", value=".005")
    body = ET.SubElement(root.find("worldbody"), "body", pos="1 0 0")
    ET.SubElement(body, "joint", name="other", axis="0 0 1", frictionloss="0")
    ET.SubElement(body, "geom", type="sphere", size=".1", mass="1", contype="0", conaffinity="0")
    sim = simulation(ET.tostring(root, encoding="unicode"))
    forces = np.zeros((1, 3, 6), dtype=np.float32)
    forces[0, 1, :3] = [3, 4, 0]
    forces[0, 2, :3] = [3, 0, 4]
    sim.data.xfrc_applied.assign(forces)
    speed(sim, [2, -2])
    sim.step(2)
    np.testing.assert_allclose(sim.loads.numpy()[0], [[3, 4], [3, 4]], atol=1e-5)
    np.testing.assert_allclose(sim.data.qfrc_constraint.numpy()[0], [-.027, .015], atol=1e-6)
    sim = simulation(xml_model(radial=0, axial=0))
    root = ET.fromstring(xml_model(radial=0, axial=0))
    root.remove(root.find("extension"))
    native = simulation(ET.tostring(root, encoding="unicode"))
    for instance in (sim, native):
        speed(instance, 2)
        set_force(instance, [3, 4, 0])
        instance.step(30)
    np.testing.assert_allclose(sim.data.qpos.numpy(), native.data.qpos.numpy(), atol=1e-7)
    np.testing.assert_allclose(sim.data.qvel.numpy(), native.data.qvel.numpy(), atol=1e-7)
    np.testing.assert_array_equal(sim.loads.numpy(), 0)


@pytest.mark.parametrize("solver", ["Newton", "CG"])
@pytest.mark.parametrize("jacobian", ["dense", "sparse"])
def test_contact_equality_native_reference(solver, jacobian):
    root = ET.fromstring(xml_model(radial=.0001, axial=.0002))
    opt = root.find("option")
    opt.set("solver", solver)
    opt.set("jacobian", jacobian)
    opt.set("gravity", "0 0 -9.81")
    world = root.find("worldbody")
    ET.SubElement(world, "geom", type="plane", size="2 2 .1")
    body = world.find("body")
    body.set("pos", "0 0 .045")
    geom = body.find("geom")
    geom.attrib.clear()
    geom.attrib.update(type="box", size=".05 .05 .05", pos=".2 0 0", mass="1")
    other = ET.SubElement(world, "body", pos="1 0 1")
    ET.SubElement(other, "joint", name="other", axis="0 1 0", frictionloss=".0003")
    ET.SubElement(other, "geom", type="sphere", size=".1", mass="1")
    ET.SubElement(ET.SubElement(root, "equality"), "joint", joint1="hinge", joint2="other")
    sim = simulation(ET.tostring(root, encoding="unicode"))
    native = sim.cpu_model
    data = mujoco.MjData(native)
    lagged = 0
    for _ in range(20):
        native.dof_frictionloss[0] = .002 + lagged
        mujoco.mj_forward(native, data)
        mujoco.mj_rnePostConstraint(native, data)
        force = data.cfrc_int[1, 3:6].copy()
        mujoco.mj_step(native, data)
        sim.step()
        np.testing.assert_allclose(sim.data.qpos.numpy()[0], data.qpos, atol=2e-5, rtol=2e-4)
        np.testing.assert_allclose(sim.data.qvel.numpy()[0], data.qvel, atol=1e-3, rtol=1e-2)
        lagged = .0001 * np.linalg.norm(force[[0, 2]]) + .0002 * abs(force[1])
    assert sim.get_data().ncon > 0
    sim.check()


def test_cuda_graph_control_changes_and_partial_reset():
    if not wp.get_device(DEVICE).is_cuda:
        pytest.skip("requires a CUDA test device")
    xml = ET.tostring(tendon_model(), encoding="unicode")
    plain, graph = simulation(xml, nworld=4), simulation(xml, nworld=4)
    for sim in (plain, graph):
        sim.step(2)
        sim.reset()
    graph.capture_step()
    np.testing.assert_array_equal(graph.data.time.numpy(), 0)
    for control in (2, -1, 0):
        for sim in (plain, graph):
            sim.set_control([control])
            sim.step(5)
        np.testing.assert_allclose(graph.loads.numpy(), plain.loads.numpy(), atol=2e-5)
        np.testing.assert_allclose(graph.limits.numpy(), plain.limits.numpy(), atol=1e-7)
        np.testing.assert_allclose(graph.data.qvel.numpy(), plain.data.qvel.numpy(), atol=2e-6)
    for sim in (plain, graph):
        sim.reset([1, 3])
        sim.set_control([2])
        sim.step(3)
    np.testing.assert_allclose(graph.loads.numpy(), plain.loads.numpy(), atol=2e-5)
    np.testing.assert_allclose(graph.data.time.numpy(), plain.data.time.numpy(), atol=1e-7)
    graph.check()


def test_includes_and_validation(tmp_path):
    root = ET.fromstring(xml_model())
    extension = root.find("extension")
    root.remove(extension)
    included = ET.Element("mujocoinclude")
    included.append(extension)
    (tmp_path / "hinges.xml").write_text(ET.tostring(included, encoding="unicode"))
    ET.SubElement(root, "include", file="hinges.xml")
    (tmp_path / "model.xml").write_text(ET.tostring(root, encoding="unicode"))
    assert len(load_model(tmp_path / "model.xml").hinges) == 1
    for key, value in [("radial_gain", "nan"), ("axial_gain", "-1"),
                       ("stribeck_velocity", "0"), ("radial_peak_width", "0")]:
        xml = xml_model().replace(f'<config key="{key}" value="0.003"/>', "")
        root = ET.fromstring(xml)
        instance = root.find("extension/plugin/instance")
        for child in list(instance):
            if child.get("key") == key:
                instance.remove(child)
        ET.SubElement(instance, "config", key=key, value=value)
        with pytest.raises(ValueError):
            load_xml(ET.tostring(root, encoding="unicode"))
    with pytest.raises(ValueError):
        load_xml(xml_model().replace('value="hinge"', 'value="missing"'))
    with pytest.raises(UnsupportedHingeError):
        load_xml(xml_model(extra='<config key="body_force_sensors" value="cable_forces"/>'))
    sim = simulation()
    with pytest.raises(IndexError):
        sim.reset([1])
    with pytest.raises(ValueError):
        sim.reset([.5])
    with pytest.raises(ValueError):
        sim.step(-1)
    with pytest.raises(ValueError):
        sim.set_control([float("nan")])


def test_repository_example_and_batch():
    sim = HingeSimulation(ROOT / "examples/hinge.xml", device=DEVICE, nworld=32)
    sim.step(20)
    sim.check()
    np.testing.assert_allclose(sim.data.qpos.numpy(), np.broadcast_to(sim.data.qpos.numpy()[0], (32, 1)), atol=1e-6)
    assert np.max(sim.loads.numpy()) > 0


def test_analytic_spin_and_stopping():
    sim = simulation(nworld=2)
    set_force(sim, [3, 4, 0])
    sim.step()
    speed(sim, [[5], [-5]])
    for step in range(400):
        sim.step()
    # I = 2*m*r^2/5 = .004. T = .027. Semi-implicit Euler
    # uses v[n+1] for q[n+1], so q differs from continuous by a*t*h/2.
    acceleration = -.027 / .004
    expected_v = 5 + acceleration * .4
    expected_q = 5 * .4 + .5 * acceleration * .4**2 + acceleration * .4 * .001 / 2
    np.testing.assert_allclose(sim.data.qvel.numpy()[:, 0], [expected_v, -expected_v], atol=5e-5)
    np.testing.assert_allclose(sim.data.qpos.numpy()[:, 0], [expected_q, -expected_q], atol=2e-5)
    sim.step(500)
    assert np.max(np.abs(sim.data.qvel.numpy())) < 2e-5
    sim.check()
