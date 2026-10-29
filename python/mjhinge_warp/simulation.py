"""Batched MuJoCo Warp simulation using native friction constraints."""

import operator
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from . import kernels as k
from .model import HingeModel, load_model


class HingeSimulation:
    """Independent worlds; loads[nworld, nhinge, 2] are radial/axial forces.

    Write data.ctrl, data.qpos, data.qvel and data.xfrc_applied on the device.
    Advance through step(), so history advances exactly once per physical step.
    Calling forward() refreshes dynamics without changing load history.
    """

    def __init__(self, source: str | Path | HingeModel, *, nworld=1,
                 device=None, nconmax=64, njmax=256):
        self.nworld = operator.index(nworld)
        nconmax, njmax = operator.index(nconmax), operator.index(njmax)
        if min(self.nworld, nconmax, njmax) < 1:
            raise ValueError("nworld, nconmax and njmax must be positive")
        self.loaded = source if isinstance(source, HingeModel) else load_model(source)
        self.cpu_model = self.loaded.model
        self.nhinge = len(self.loaded.hinges)
        self.device = wp.get_device(device)
        self._graph = None
        with wp.ScopedDevice(self.device):
            self.model = mjw.put_model(self.cpu_model)
            # Reserve sparse storage for zero-base hinges that acquire friction.
            self.data = mjw.make_data(self.cpu_model, nworld=self.nworld,
                                      nconmax=nconmax, njmax=njmax,
                                      njmax_nnz=njmax * max(1, self.cpu_model.nv))
            parameters = []
            row_seeds = self.cpu_model.dof_frictionloss.copy()[None, :]
            for spec in self.loaded.hinges:
                p = k.Parameters()
                for name in ("joint", "dof", "body", "qpos"):
                    setattr(p, name, getattr(spec, name))
                p.active = int(spec.active)
                p.base = float(self.cpu_model.dof_frictionloss[spec.dof])
                for name, value in spec.config.items():
                    setattr(p, name, value)
                parameters.append(p)
                if spec.active:
                    # Keep a row available in every RK stage/world, including
                    # first step and unload. The callback replaces its exact
                    # limit (possibly zero) before the native solver runs.
                    row_seeds[0, spec.dof] = max(p.base, 1e-30)
            self.parameters = wp.array(parameters, dtype=k.Parameters)
            self.loads = wp.zeros((self.nworld, self.nhinge), dtype=wp.vec2)
            self.limits = wp.zeros((self.nworld, self.nhinge), dtype=float)
            if self.nhinge:
                self.model.dof_frictionloss = wp.array(row_seeds, dtype=float)
                self.model.callback.passive = self._passive
                # RNE runs after each solve, BEFORE integration changes qvel;
                # 3.16 already includes spatial tendon forces. Do not add the
                # MuJoCo 3.9 C++ backend's tendon correction a second time.
                self.model.opt.run_rne_postconstraint = True
            self.forward()

    def _enabled(self):
        flags = int(self.model.opt.disableflags)
        bit = mujoco.mjtDisableBit
        return not (flags & (bit.mjDSBL_CONSTRAINT | bit.mjDSBL_FRICTIONLOSS)) and not (
            flags & bit.mjDSBL_SPRING and flags & bit.mjDSBL_DAMPER)

    def _passive(self, model, data):
        wp.launch(k.limits, (self.nworld, self.nhinge), inputs=[
            self.parameters, self.loads, data.qpos, data.qvel, int(self._enabled()),
            self.limits], device=self.device)
        wp.launch(k.update_rows, (self.nworld, data.njmax), inputs=[
            self.parameters, self.limits, data.nefc, data.efc.type, data.efc.id,
            data.efc.frictionloss], device=self.device)

    def forward(self):
        with wp.ScopedDevice(self.device):
            mjw.forward(self.model, self.data)

    def _step(self):
        mjw.step(self.model, self.data)
        if self.nhinge and self._enabled():
            wp.launch(k.advance, (self.nworld, self.nhinge), inputs=[
                self.parameters, self.data.cfrc_int, self.data.xaxis, self.loads],
                device=self.device)

    def step(self, count=1):
        count = operator.index(count)
        if count < 0:
            raise ValueError("step count must be nonnegative")
        with wp.ScopedDevice(self.device):
            for _ in range(count):
                if self._graph is None:
                    self._step()
                else:
                    wp.capture_launch(self._graph)

    def capture_step(self):
        """Capture one step after warm-up; capture itself does not advance CUDA.

        After capture, update array contents in place. Changing callbacks,
        options or topology requires clear_graph() and a new capture.
        """
        if not self.device.is_cuda:
            raise ValueError("CUDA graph capture requires a CUDA device")
        with wp.ScopedDevice(self.device), wp.ScopedCapture() as capture:
            self._step()
        self._graph = capture.graph

    def clear_graph(self):
        self._graph = None

    def reset(self, worlds=None):
        mask = np.zeros(self.nworld, dtype=bool)
        if worlds is None:
            mask[:] = True
        else:
            indices = np.asarray(worlds)
            if indices.size and indices.dtype.kind not in "iu":
                raise ValueError("world indices must be integers")
            indices = indices.astype(np.int64).reshape(-1)
            if np.any(indices < 0) or np.any(indices >= self.nworld):
                raise IndexError("world index out of range")
            mask[indices] = True
        with wp.ScopedDevice(self.device):
            active = wp.array(mask, dtype=bool)
            mjw.reset_data(self.model, self.data, reset=active)
            if self.nhinge:
                wp.launch(k.reset_history, (self.nworld, self.nhinge),
                          inputs=[active, self.loads], device=self.device)
            self.forward()

    def set_control(self, controls):
        controls = np.asarray(controls, dtype=np.float32)
        shape = (self.nworld, self.cpu_model.nu)
        if controls.shape == (self.cpu_model.nu,):
            controls = np.broadcast_to(controls, shape)
        if controls.shape != shape or not np.all(np.isfinite(controls)):
            raise ValueError(f"controls must be finite with shape {shape}")
        self.data.ctrl.assign(np.ascontiguousarray(controls))

    def get_data(self, world=0, result=None):
        world = operator.index(world)
        if not 0 <= world < self.nworld:
            raise IndexError("world index out of range")
        if result is None:
            result = mujoco.MjData(self.cpu_model)
        mjw.get_data_into(result, self.cpu_model, self.data, world_id=world)
        return result

    def check(self):
        """Synchronize explicitly to check finite state and buffer capacities."""
        for array in (self.data.qpos, self.data.qvel, self.data.qacc,
                      self.data.qfrc_constraint, self.loads, self.limits):
            if not np.all(np.isfinite(array.numpy())):
                raise RuntimeError("Non-finite hinge simulation state")
        if np.any(self.data.overflow.numpy()):
            raise RuntimeError("MJWarp buffer overflow; increase nconmax/njmax")
