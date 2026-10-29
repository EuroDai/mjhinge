"""Load-dependent hinge friction on CPU/CUDA, without a C++ plugin binary."""

from .model import HingeModel, HingeSpec, UnsupportedHingeError, load_model, load_xml

__all__ = ["HingeSimulation", "HingeModel", "HingeSpec", "UnsupportedHingeError", "load_model", "load_xml"]


def __getattr__(name):
    if name == "HingeSimulation":
        from .simulation import HingeSimulation
        return HingeSimulation
    raise AttributeError(name)
