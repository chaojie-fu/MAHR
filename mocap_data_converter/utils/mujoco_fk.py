"""MuJoCo-backed forward kinematics helpers for URDF robots.

This module provides a wrapper around :mod:`urdfpy` robots that swaps the
numeric forward-kinematics implementation with MuJoCo while keeping the rest of
``urdfpy``'s convenient metadata (joint limits, names, etc.).

The wrapper automatically falls back to ``urdfpy``'s implementation when
symbolic inputs are detected so that the existing symbolic optimisation pipeline
continues to function without any modifications.
"""
from __future__ import annotations

import time
import logging
from typing import Dict, Iterable, Optional

import numpy as np
import sympy

try:  # pragma: no cover - optional dependency is handled at runtime
    import mujoco
except ImportError as exc:  # pragma: no cover - informative error at runtime
    raise ImportError(
        "The 'mujoco' package is required when using the MuJoCo forward "
        "kinematics backend. Please install it via 'pip install mujoco'."
    ) from exc

from urdfpy import URDF

__all__ = ["MujocoURDFRobot"]


def _iter_values(value) -> Iterable:
    """Yield leaf values from nested containers."""
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_values(item)
    elif isinstance(value, (list, tuple, np.ndarray)):
        for item in value:
            yield from _iter_values(item)
    else:
        yield value


def _contains_symbolic(cfg) -> bool:
    """Return ``True`` when *cfg* contains symbolic (SymPy) values."""
    if cfg is None:
        return False
    return any(isinstance(v, sympy.Basic) for v in _iter_values(cfg))


class MujocoURDFRobot:
    """Hybrid URDF robot whose numeric FK is delegated to MuJoCo."""

    def __init__(self, urdf_path: str):
        self._urdf: URDF = URDF.load(urdf_path)
        self._model = mujoco.MjModel.from_xml_path(urdf_path)
        self._data = mujoco.MjData(self._model)

        # Cache joint -> qpos mapping for quick assignment.
        self._joint_qpos_index: Dict[str, int] = {}
        for joint_name in self._urdf.actuated_joint_names:
            joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                raise KeyError(
                    f"Joint '{joint_name}' defined in the URDF could not be found in the MuJoCo model."
                )
            self._joint_qpos_index[joint_name] = int(self._model.jnt_qposadr[joint_id])

        # Cache body lookup for URDF link names.
        self._body_name_to_id: Dict[str, int] = {}
        for body_id in range(self._model.nbody):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name:
                self._body_name_to_id[name] = body_id

        self.fk_dict = {}
        for link in self._urdf.links:
            assert body_id is not None, f"Link name {link.name} not found in MuJoCo model."
            transform = np.eye(4, dtype=np.float64)
            self.fk_dict[link.name] = transform

    # ------------------------------------------------------------------
    # ``urdfpy`` compatibility helpers
    # ------------------------------------------------------------------
    def __getattr__(self, item):
        return getattr(self._urdf, item)

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------
    def link_fk(
        self,
        cfg=None,
        link: Optional[str] = None,
        links: Optional[Iterable[str]] = None,
        use_names: bool = False,
    ):
        """Compute link poses using MuJoCo when possible.

        When symbolic values are supplied this method falls back to the native
        ``urdfpy`` implementation to keep the optimisation pipeline intact.
        """
        qpos = np.zeros(self._model.nq, dtype=float)
        if cfg is not None:
            if isinstance(cfg, dict):
                for joint_name, value in cfg.items():
                    if joint_name not in self._joint_qpos_index:
                        continue
                    qpos[self._joint_qpos_index[joint_name]] = float(value)
            else:
                cfg_values = np.asarray(cfg, dtype=float)
                for idx, joint_name in enumerate(self._urdf.actuated_joint_names):
                    if idx >= len(cfg_values):
                        break
                    qpos[self._joint_qpos_index[joint_name]] = float(cfg_values[idx])

        # Reset model state before evaluation to avoid accumulating drift from
        # previous calls.
        self._data.qpos[:] = qpos
        self._data.qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)

        for link in self._urdf.links:
            body_id = self._body_name_to_id.get(link.name)
            assert body_id is not None, f"Link name {link.name} not found in MuJoCo model."
            self.fk_dict[link.name][:3, :3] = self._data.xmat[body_id].reshape(3, 3)
            self.fk_dict[link.name][:3, 3] = self._data.xpos[body_id]

        return self.fk_dict
