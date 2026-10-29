"""Read existing mjhinge MJCF without loading its native plugin."""

from dataclasses import dataclass
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import mujoco

PLUGIN = "mjhinge.hinge_friction"
DEFAULTS = dict(radial_gain=0.0, axial_gain=0.0, static_extra=0.0,
                stribeck_velocity=0.02, radial_peak_gain=0.0,
                radial_peak_angle=0.7, radial_peak_width=0.2)


class UnsupportedHingeError(ValueError):
    """A requested feature cannot be represented by this backend."""


@dataclass(frozen=True)
class HingeSpec:
    name: str
    joint: int
    dof: int
    body: int
    qpos: int
    config: dict[str, float]

    @property
    def active(self):
        return any(self.config[k] for k in
                   ("radial_gain", "axial_gain", "static_extra", "radial_peak_gain"))


@dataclass
class HingeModel:
    model: mujoco.MjModel
    hinges: list[HingeSpec]


def _includes(root, directory, stack):
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag != "include":
                continue
            path = (directory / child.attrib["file"]).resolve()
            if path in stack:
                raise ValueError(f"Recursive MJCF include: {path}")
            included = ET.parse(path).getroot()
            _includes(included, path.parent, (*stack, path))
            index = list(parent).index(child)
            parent.remove(child)
            for offset, element in enumerate(included):
                parent.insert(index + offset, element)


def load_model(path: str | Path) -> HingeModel:
    path = Path(path).resolve()
    return load_xml(path.read_text(encoding="utf-8"), base_dir=path.parent)


def load_xml(xml: str, *, base_dir: str | Path = ".") -> HingeModel:
    """Compile a plugin-free copy; includes and assets resolve from base_dir."""
    directory = Path(base_dir).resolve()
    root = ET.fromstring(xml)
    _includes(root, directory, ())
    configurations = []
    names = set()
    for extension in root.findall("extension"):
        for plugin in list(extension):
            if plugin.tag != "plugin" or plugin.get("plugin") != PLUGIN:
                raise UnsupportedHingeError("Foreign engine plugins require their own MJWarp backend")
            for instance in plugin:
                if instance.tag != "instance":
                    raise ValueError("Expected a named mjhinge instance")
                name = instance.get("name", "")
                if not name or name in names:
                    raise ValueError("Hinge instance names must be nonempty and unique")
                names.add(name)
                config = {}
                for item in instance:
                    key = item.get("key")
                    if item.tag != "config" or key in config:
                        raise ValueError(f"{name}: invalid or duplicate config")
                    config[key] = item.get("value", "")
                unknown = set(config) - set(DEFAULTS) - {"joint", "body_force_sensors"}
                if unknown:
                    raise ValueError(f"{name}: unknown options: {sorted(unknown)}")
                if config.get("body_force_sensors", "").strip():
                    raise UnsupportedHingeError(f"{name}: body_force_sensors requires an external cable backend; use physical xfrc_applied or native tendons")
                joint = config.get("joint", "").strip()
                if not joint:
                    raise ValueError(f"{name}: joint is required")
                values = {}
                for key, default in DEFAULTS.items():
                    text = config.get(key, "")
                    value = float(default if text == "" else text)
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f"{name}: {key} must be finite and nonnegative")
                    if key in ("stribeck_velocity", "radial_peak_width") and value <= 0:
                        raise ValueError(f"{name}: {key} must be positive")
                    values[key] = value
                configurations.append((name, joint, values))
            extension.remove(plugin)
        root.remove(extension)
    if root.findall(".//plugin"):
        raise UnsupportedHingeError("Inline actuator/body/sensor plugins are not supported")
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    assetdir = compiler.attrib.pop("assetdir", "")
    for key in ("meshdir", "texturedir"):
        compiler.set(key, str((directory / compiler.get(key, assetdir)).resolve()))
    for asset in root.findall("asset/*"):
        if asset.get("file") and asset.tag not in ("mesh", "texture"):
            asset.set("file", str((directory / asset.get("file")).resolve()))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP:
        raise UnsupportedHingeError("Sleeping bodies are not supported by the hinge backend")
    hinges = []
    used = set()
    for name, joint_name, config in configurations:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"{name}: {joint_name!r} must name a hinge joint")
        body = int(model.jnt_bodyid[joint])
        if model.body_jntnum[body] != 1 or joint in used:
            raise ValueError(f"{name}: hinge must be the body's only joint and have one instance")
        used.add(joint)
        hinges.append(HingeSpec(name, joint, int(model.jnt_dofadr[joint]), body,
                                int(model.jnt_qposadr[joint]), config))
    return HingeModel(model, hinges)
