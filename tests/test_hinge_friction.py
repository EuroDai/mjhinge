"""Behavioral tests for the standalone plugin; requires numpy and mujoco."""

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def xml_model(base=0.002, radial=0.003, axial=0.004, axis="0 1 0",
              quat="1 0 0 0", integrator="Euler", extra_config=""):
    config = ""
    for key, value in (("radial_gain", radial), ("axial_gain", axial)):
        if value is not None:
            config += f'<config key="{key}" value="{value}"/>'
    return f"""<mujoco>
      <option gravity="0 0 0" timestep="0.001" integrator="{integrator}"
              iterations="100" tolerance="1e-12"/>
      <extension><plugin plugin="mjhinge.hinge_friction"><instance name="f">
        <config key="joint" value="hinge"/>{config}{extra_config}
      </instance></plugin></extension>
      <worldbody><body name="link" quat="{quat}">
        <joint name="hinge" axis="{axis}" frictionloss="{base}"
               solreffriction="0.002 1" solimpfriction="0.9999 0.9999 0.001"/>
        <geom type="sphere" size="0.1" mass="1" contype="0" conaffinity="0"/>
      </body></worldbody>
    </mujoco>"""


def without_plugin(xml):
    root = ET.fromstring(xml)
    root.remove(root.find("extension"))
    return ET.tostring(root, encoding="unicode")


def make(xml):
    model = mujoco.MjModel.from_xml_string(xml)
    return model, mujoco.MjData(model)


def limit(data, dof=0):
    rows = np.flatnonzero(
        (data.efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
        & (data.efc_id == dof)
    )
    return float(data.efc_frictionloss[rows[0]]) if len(rows) else 0.0


class HingeFrictionTest(unittest.TestCase):
    def test_angle_dependent_bearing_friction(self):
        config = ('<config key="radial_peak_gain" value=".003"/>'
                  '<config key="radial_peak_angle" value=".7"/>'
                  '<config key="radial_peak_width" value=".2"/>')
        for angle in [0., .7, 1.4]:
            m,d=make(xml_model(base=0,radial=0,axial=0,extra_config=config))
            d.qpos[0]=angle
            d.plugin_state[:]=[2.,0.]
            mujoco.mj_forward(m,d)
            self.assertAlmostEqual(limit(d), .006*np.exp(-.5*((angle-.7)/.2)**2))

    def test_stribeck_static_extra_and_sliding_dissipation(self):
        config = '<config key="static_extra" value=".01"/><config key="stribeck_velocity" value=".02"/>'
        for base in [0, .002]:
            m, d = make(xml_model(base=base, radial=0, axial=0, extra_config=config))
            d.qfrc_applied[0] = .005
            mujoco.mj_step(m, d, 100)
            self.assertLess(abs(d.qpos[0]), 1e-4)
            self.assertAlmostEqual(limit(d), base+.01, places=6)
            for speed in [-1., 1.]:
                mujoco.mj_resetData(m, d)
                d.qvel[0] = speed
                mujoco.mj_step(m, d)
                self.assertAlmostEqual(limit(d), base, places=10)
                self.assertLessEqual(d.qfrc_constraint[0]*speed, 1e-12)
        with self.assertRaises((ValueError, mujoco.FatalError)):
            make(xml_model(extra_config='<config key="stribeck_velocity" value="0"/>'))

    def test_spatial_tendon_site_forces_and_pulley(self):
        for clamped in (False, True):
            root = ET.fromstring(xml_model(base=0))
            world = root.find('worldbody')
            ET.SubElement(world, 'site', name='a', pos='.3 .4 0')
            ET.SubElement(world, 'site', name='b', pos='-.3 .4 0')
            ET.SubElement(world.find('body'), 'site', name='centre', pos='0 0 0')
            spatial = ET.SubElement(ET.SubElement(root, 'tendon'), 'spatial', name='cable')
            if clamped:
                spatial.set('actuatorfrclimited', 'true')
                spatial.set('actuatorfrcrange', '-1 1')
            for tag, attrs in [('site', dict(site='a')), ('site', dict(site='centre')),
                               ('pulley', dict(divisor='2')), ('site', dict(site='b')),
                               ('site', dict(site='centre'))]:
                ET.SubElement(spatial, tag, **attrs)
            ET.SubElement(ET.SubElement(root, 'actuator'), 'motor', tendon='cable', gear='3')
            xml = ET.tostring(root, encoding='unicode')
            for ctrl in (-2, 2):
                m, d = make(xml)
                d.ctrl[0] = ctrl
                mujoco.mj_step(m, d)
                # Each segment pulls along a 3-4-5 triangle. Second branch has
                # half tension. Both attach at the fixed COM: no acceleration.
                expected = np.array([1.8, 7.2]) * (.5 if clamped else 1)
                np.testing.assert_allclose(d.plugin_state, expected, atol=1e-12)
                mujoco.mj_step(m, d)
                self.assertAlmostEqual(limit(d), expected @ [0.003, 0.004])
                m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
                mujoco.mj_step(m, d)
                np.testing.assert_allclose(d.plugin_state, 0, atol=1e-12)
            spatial.set('stiffness', '1')
            with self.assertRaises((ValueError, mujoco.FatalError)):
                make(ET.tostring(root, encoding='unicode'))

    def test_one_step_delay_and_load_changes(self):
        m, d = make(xml_model())
        d.qvel[0] = 5
        d.xfrc_applied[1, :3] = (3, 4, 0)
        mujoco.mj_step(m, d)
        self.assertAlmostEqual(limit(d), 0.002)
        np.testing.assert_allclose(d.plugin_state, [3, 4], atol=1e-12)
        mujoco.mj_step(m, d)
        self.assertAlmostEqual(limit(d), 0.027)
        self.assertAlmostEqual(d.qfrc_constraint[0], -0.027)
        d.xfrc_applied[1, :3] = (0, -10, 0)
        mujoco.mj_step(m, d)
        self.assertAlmostEqual(limit(d), 0.027)
        mujoco.mj_step(m, d)
        self.assertAlmostEqual(limit(d), 0.042)
        d.xfrc_applied[:] = 0
        mujoco.mj_step(m, d)
        self.assertAlmostEqual(limit(d), 0.042)
        mujoco.mj_step(m, d)
        self.assertAlmostEqual(limit(d), 0.002)
        np.testing.assert_array_equal(m.dof_frictionloss, [0.002])

    def test_radial_and_axial_independently(self):
        for force, expected, loads in [
            ((3, 0, 4), 0.017, (5, 0)),
            ((0, -4, 0), 0.018, (0, 4)),
            ((3, 4, 0), 0.027, (3, 4)),
        ]:
            with self.subTest(force=force):
                m, d = make(xml_model())
                d.qvel[0] = 5
                d.xfrc_applied[1, :3] = force
                mujoco.mj_step(m, d, 2)
                self.assertAlmostEqual(limit(d), expected)
                np.testing.assert_allclose(d.plugin_state, loads, atol=1e-12)

    def test_zero_frictionloss_and_integrators(self):
        for integrator in ("Euler", "implicit", "implicitfast", "RK4"):
            for base in (0, 0.002):
                with self.subTest(integrator=integrator, base=base):
                    m, d = make(xml_model(base=base, integrator=integrator))
                    d.qvel[0] = -5
                    d.xfrc_applied[1, :3] = (3, 4, 0)
                    mujoco.mj_step(m, d, 3)
                    self.assertAlmostEqual(limit(d), base + 0.025)
                    self.assertAlmostEqual(d.qfrc_constraint[0], base + 0.025)
                    self.assertEqual(m.dof_frictionloss[0], base)

    def test_zero_gains_match_native_friction(self):
        xml = xml_model(radial=0, axial=0)
        m, d = make(xml)
        native, reference = make(without_plugin(xml))
        for data in (d, reference):
            data.qvel[0] = 2
            data.qfrc_applied[0] = 0.001
            data.xfrc_applied[1, :3] = (3, 4, 5)
        for _ in range(100):
            mujoco.mj_step(m, d)
            mujoco.mj_step(native, reference)
            np.testing.assert_array_equal(d.qpos, reference.qpos)
            np.testing.assert_array_equal(d.qvel, reference.qvel)
            np.testing.assert_array_equal(d.qfrc_constraint, reference.qfrc_constraint)
        np.testing.assert_array_equal(d.plugin_state, [0, 0])

    def test_static_hold_and_breakaway(self):
        m, d = make(xml_model(base=0))
        d.xfrc_applied[1, :3] = (3, 4, 0)
        d.qfrc_applied[0] = 0.01  # below the load-dependent limit of 0.025
        mujoco.mj_step(m, d, 200)
        self.assertLess(abs(d.qvel[0]), 1e-5)
        self.assertAlmostEqual(d.qfrc_constraint[0], -0.01, places=6)
        d.qfrc_applied[0] = 0.05
        mujoco.mj_step(m, d, 100)
        self.assertGreater(d.qvel[0], 0.1)
        self.assertAlmostEqual(d.qfrc_constraint[0], -0.025)

    def test_world_axis_and_body_rotation(self):
        # Local x becomes world y after a 90 degree rotation about z.
        m, d = make(xml_model(axis="1 0 0", quat="1 0 0 1"))
        d.xfrc_applied[1, :3] = (3, 4, 0)
        mujoco.mj_step(m, d, 2)
        np.testing.assert_allclose(d.plugin_state, [3, 4], atol=1e-12)
        self.assertAlmostEqual(limit(d), 0.027)

    def test_gravity_and_subtree_reaction(self):
        root = ET.fromstring(xml_model())
        root.find("option").set("gravity", "0 0 -10")
        child = ET.SubElement(root.find("worldbody/body"), "body")
        ET.SubElement(child, "geom", type="sphere", size="0.05", mass="0.5",
                      contype="0", conaffinity="0")
        m, d = make(ET.tostring(root, encoding="unicode"))
        mujoco.mj_step(m, d, 2)
        np.testing.assert_allclose(d.plugin_state, [15, 0], atol=1e-12)
        self.assertAlmostEqual(limit(d), 0.047)

    def test_reaction_uses_velocity_before_integration(self):
        root = ET.fromstring(xml_model(axis="0 0 1", radial=0.001))
        root.find("worldbody/body/geom").set("pos", "0.3 0 0")
        m, d = make(ET.tostring(root, encoding="unicode"))
        d.qvel[0] = 2
        mujoco.mj_forward(m, d)
        mujoco.mj_rnePostConstraint(m, d)
        expected = np.linalg.norm(d.cfrc_int[1, 3:5])
        mujoco.mj_step(m, d)
        self.assertAlmostEqual(d.plugin_state[0], expected, places=12)

    def test_multiple_instances_and_rows(self):
        root = ET.fromstring(xml_model(base=0))
        plugin = root.find("extension/plugin")
        second = ET.SubElement(plugin, "instance", name="second")
        ET.SubElement(second, "config", key="joint", value="other")
        ET.SubElement(second, "config", key="radial_gain", value="0.005")
        ET.SubElement(second, "config", key="axial_gain", value="0.006")
        body = copy.deepcopy(root.find("worldbody/body"))
        body.set("name", "other_link")
        body.set("pos", "1 0 0")
        body.find("joint").set("name", "other")
        body.find("joint").set("axis", "0 0 1")
        body.find("joint").set("frictionloss", "0.003")
        root.find("worldbody").append(body)
        m, d = make(ET.tostring(root, encoding="unicode"))
        d.qvel[:] = (5, -5)
        d.xfrc_applied[1, :3] = (3, 4, 0)
        d.xfrc_applied[2, :3] = (0, 3, 4)
        mujoco.mj_step(m, d, 2)
        np.testing.assert_allclose(d.plugin_state, [3, 4, 3, 4], atol=1e-12)
        self.assertAlmostEqual(limit(d, 0), 0.025)
        self.assertAlmostEqual(limit(d, 1), 0.042)
        np.testing.assert_allclose(d.qfrc_constraint, [-0.025, 0.042], atol=1e-12)
        np.testing.assert_array_equal(m.dof_frictionloss, [0, 0.003])

    def test_reset_copy_and_state_replay(self):
        m, d = make(xml_model())
        d.qvel[0] = 2
        d.xfrc_applied[1, :3] = (3, 4, 0)
        mujoco.mj_step(m, d, 5)
        cloned = copy.copy(d)
        for _ in range(5):
            mujoco.mj_step(m, d)
            mujoco.mj_step(m, cloned)
            np.testing.assert_array_equal(d.qpos, cloned.qpos)
            np.testing.assert_array_equal(d.plugin_state, cloned.plugin_state)

        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(m, spec))
        mujoco.mj_getState(m, d, state, spec)
        replay = mujoco.MjData(m)
        mujoco.mj_setState(m, replay, state, spec)
        mujoco.mj_step(m, d)
        mujoco.mj_step(m, replay)
        np.testing.assert_array_equal(d.qpos, replay.qpos)
        np.testing.assert_array_equal(d.plugin_state, replay.plugin_state)

        mujoco.mj_resetData(m, cloned)
        np.testing.assert_array_equal(cloned.plugin_state, [0, 0])
        cloned.qvel[0] = 2
        mujoco.mj_step(m, cloned)
        self.assertAlmostEqual(limit(cloned), 0.002)
        np.testing.assert_allclose(d.plugin_state, [3, 4], atol=1e-12)

    def test_forward_queries_do_not_advance_history(self):
        m, d = make(xml_model())
        d.xfrc_applied[1, :3] = (3, 4, 0)
        for _ in range(5):
            mujoco.mj_forward(m, d)
        np.testing.assert_array_equal(d.plugin_state, [0, 0])
        mujoco.mj_step(m, d)
        saved = d.plugin_state.copy()
        d.xfrc_applied[:] = 0
        for _ in range(5):
            mujoco.mj_forward(m, d)
        np.testing.assert_array_equal(d.plugin_state, saved)
        self.assertAlmostEqual(limit(d), 0.027)

    def test_parallel_data_share_model_safely(self):
        m = mujoco.MjModel.from_xml_string(xml_model(base=0))

        def rollout(scale):
            d = mujoco.MjData(m)
            d.qvel[0] = 5
            d.xfrc_applied[1, :3] = (scale, 2 * scale, 0)
            mujoco.mj_step(m, d, 50)
            return np.r_[d.qpos, d.qvel, d.plugin_state, limit(d)]

        scales = [1, 2, 3, 4]
        expected = [rollout(scale) for scale in scales]
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(rollout, scales))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(m.dof_frictionloss, [0])

    def test_split_step_and_disable_flags(self):
        m, d = make(xml_model(base=0))
        d.qvel[0] = 5
        d.xfrc_applied[1, :3] = (3, 4, 0)
        for _ in range(3):
            mujoco.mj_step1(m, d)
            mujoco.mj_step2(m, d)
        self.assertAlmostEqual(limit(d), 0.025)
        for flag in (mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS,
                     mujoco.mjtDisableBit.mjDSBL_CONSTRAINT):
            m.opt.disableflags = flag
            mujoco.mj_step(m, d)
            self.assertEqual(limit(d), 0)
            self.assertEqual(d.qfrc_constraint[0], 0)
        m.opt.disableflags = 0
        mujoco.mj_step(m, d)
        self.assertAlmostEqual(limit(d), 0.025)

    def test_contact_and_equality_match_native_solver(self):
        # Compare whole trajectories to native frictionloss updated by a
        # reference harness. The plugin host itself only calls mj_step.
        for solver in ("Newton", "CG", "PGS"):
            for jacobian in ("dense", "sparse"):
                for base in (0, 0.002):
                    with self.subTest(solver=solver, jacobian=jacobian, base=base):
                        root = ET.fromstring(xml_model(base=base, radial=0.0001,
                                                       axial=0.0002))
                        option = root.find("option")
                        option.set("solver", solver)
                        option.set("jacobian", jacobian)
                        option.set("gravity", "0 0 -9.81")
                        world = root.find("worldbody")
                        ET.SubElement(world, "geom", type="plane", size="2 2 0.1")
                        body = world.find("body")
                        body.set("pos", "0 0 0.045")
                        geom = body.find("geom")
                        geom.attrib.clear()
                        geom.attrib.update(type="box", size="0.05 0.05 0.05",
                                           pos="0.2 0 0", mass="1")
                        ET.SubElement(body, "site", name="reaction", size="0.005")
                        other = ET.SubElement(world, "body", pos="1 0 1")
                        ET.SubElement(other, "joint", name="other", axis="0 1 0",
                                      frictionloss="0.0003")
                        ET.SubElement(other, "geom", type="sphere", size="0.1", mass="1")
                        equality = ET.SubElement(root, "equality")
                        ET.SubElement(equality, "joint", joint1="hinge", joint2="other")
                        sensor = ET.SubElement(root, "sensor")
                        ET.SubElement(sensor, "force", site="reaction")
                        xml = ET.tostring(root, encoding="unicode")
                        m, d = make(xml)
                        native, reference = make(without_plugin(xml))
                        lagged = 0
                        for _ in range(20):
                            native.dof_frictionloss[0] = base + lagged
                            mujoco.mj_step(m, d)
                            mujoco.mj_step(native, reference)
                            np.testing.assert_allclose(d.qpos, reference.qpos,
                                                       rtol=1e-9, atol=1e-11)
                            np.testing.assert_allclose(d.qvel, reference.qvel,
                                                       rtol=1e-8, atol=1e-10)
                            force = reference.cfrc_int[1, 3:6]
                            lagged = 0.0001 * np.linalg.norm(force[[0, 2]]) + 0.0002 * abs(force[1])
                            np.testing.assert_allclose(
                                d.plugin_state, [np.linalg.norm(force[[0, 2]]), abs(force[1])],
                                rtol=1e-8, atol=1e-10)
                        self.assertGreater(d.ncon, 0)
                        self.assertEqual(m.dof_frictionloss[0], base)

    def test_configuration_validation(self):
        for bad in ("-1", "nan", "inf", "1mm", "1e999", " "):
            for key in ("radial_gain", "axial_gain"):
                with self.subTest(key=key, value=bad):
                    kwargs = {"radial" if key == "radial_gain" else "axial": bad}
                    with self.assertRaises((ValueError, mujoco.FatalError)):
                        make(xml_model(**kwargs))
        m, d = make(xml_model(radial=None, axial=None))
        mujoco.mj_step(m, d)
        self.assertAlmostEqual(limit(d), 0.002)
        for joint in ("", "missing"):
            with self.subTest(joint=joint):
                xml = xml_model().replace('key="joint" value="hinge"',
                                           f'key="joint" value="{joint}"')
                with self.assertRaises((ValueError, mujoco.FatalError)):
                    make(xml)
        for bad_type in ("slide", "ball"):
            root = ET.fromstring(xml_model())
            root.find("worldbody/body/joint").set("type", bad_type)
            with self.assertRaises((ValueError, mujoco.FatalError)):
                make(ET.tostring(root, encoding="unicode"))

        root = ET.fromstring(xml_model())
        plugin = root.find("extension/plugin")
        duplicate = copy.deepcopy(plugin.find("instance"))
        duplicate.set("name", "duplicate")
        plugin.append(duplicate)
        with self.assertRaises((ValueError, mujoco.FatalError)):
            make(ET.tostring(root, encoding="unicode"))
        root = ET.fromstring(xml_model())
        ET.SubElement(root.find("worldbody/body"), "joint", name="second", axis="1 0 0")
        with self.assertRaises((ValueError, mujoco.FatalError)):
            make(ET.tostring(root, encoding="unicode"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", type=Path, required=True)
    args, test_args = parser.parse_known_args()
    mujoco.mj_loadPluginLibrary(str(args.plugin.resolve()))
    unittest.main(argv=[__file__, *test_args])
