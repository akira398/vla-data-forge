"""
Base schema primitives shared across all dataset types.

Design notes
------------
- We use plain Python dataclasses (not Pydantic) for in-memory data objects because
  batch processing of thousands of episodes benefits from lightweight construction.
  Pydantic validation overhead adds up at scale.
- Numpy arrays are stored directly on dataclass fields.  When an object needs to be
  serialised (JSONL export), the export layer handles the numpy→list conversion.
- ``RobotAction`` wraps the standard 7-DoF action vector used by both Bridge v2 and
  ECoT.  Named fields make downstream code self-documenting.
- ``NumpyArrayMixin`` is a lightweight helper for any dataclass that needs array I/O
  helpers — it is not a base class to inherit blindly, just a useful tool.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RobotAction:
    """
    Standard 7-DoF robot end-effector action.

    Convention (matches Bridge v2 and most BerkeleyUR5 datasets):
        [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
    Gripper: 0.0 = fully open, 1.0 = fully closed.
    """

    delta_x: float = 0.0
    delta_y: float = 0.0
    delta_z: float = 0.0
    delta_roll: float = 0.0
    delta_pitch: float = 0.0
    delta_yaw: float = 0.0
    gripper: float = 0.0

    DIM: int = field(default=7, init=False, repr=False)

    def to_numpy(self) -> np.ndarray:
        """Return (7,) float32 array."""
        return np.array(
            [
                self.delta_x,
                self.delta_y,
                self.delta_z,
                self.delta_roll,
                self.delta_pitch,
                self.delta_yaw,
                self.gripper,
            ],
            dtype=np.float32,
        )

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "RobotAction":
        """Construct from a (7,) array.  Raises ValueError if shape is wrong."""
        arr = np.asarray(arr, dtype=np.float32).flatten()
        if arr.shape[0] != 7:
            raise ValueError(f"Expected 7-element action vector, got shape {arr.shape}")
        return cls(
            delta_x=float(arr[0]),
            delta_y=float(arr[1]),
            delta_z=float(arr[2]),
            delta_roll=float(arr[3]),
            delta_pitch=float(arr[4]),
            delta_yaw=float(arr[5]),
            gripper=float(arr[6]),
        )

    def to_list(self) -> List[float]:
        return self.to_numpy().tolist()


class NumpyArrayMixin:
    """
    Mixin helpers for classes that hold numpy arrays.

    Call ``_numpy_to_list(self)`` before JSON serialisation to convert all
    ndarray fields to nested Python lists.  This is intentionally explicit so
    that callers decide when to pay the conversion cost.
    """

    @staticmethod
    def array_to_list(arr: Optional[np.ndarray]) -> Optional[List]:
        """Convert ndarray to nested list, or pass through None."""
        if arr is None:
            return None
        return arr.tolist()

    @staticmethod
    def list_to_array(lst: Optional[List], dtype: str = "float32") -> Optional[np.ndarray]:
        """Convert nested list back to ndarray."""
        if lst is None:
            return None
        return np.array(lst, dtype=dtype)


def safe_numpy_equal(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> bool:
    """Equality check that handles None and arrays cleanly."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return np.array_equal(a, b)
