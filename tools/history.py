"""Undo/redo history for viewer edits (tools/history).

A tiny, deliberately generic stack so every viewer-side edit kind -
smooth-brush strokes, lasso selections, polygroup painting - joins the
same Ctrl+Z flow without new plumbing.  An entry is any object with:

* ``label``  - short human-readable name shown in logs/menus
* ``undo()`` - restore the previous state; True when it was applied
* ``redo()`` - re-apply it; True when it was applied

Concrete kinds:

* :class:`VertexEditEntry` - snapshots the positions (and normals) of
  exactly the vertices a smooth-brush stroke touched, so undoing is an
  indexed scatter - no full-mesh copies.
* :class:`StateEntry`      - the generic kind for snapshot states: the
  caller hands in an applier callable plus the state before and after the
  edit (dicts of per-actor arrays for lasso selections and polygroup
  assignments).  The applier returns False when the state no longer fits
  (e.g. the mesh was reloaded and its actors are gone), which fails the
  undo/redo without corrupting anything.

The stack is LIFO with a redo tail, entries are dropped oldest-first past
``limit`` (each entry only holds the state it touched, so memory stays
proportional to the edits, not the mesh).
"""

from __future__ import annotations

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

_DEFAULT_LIMIT = 40


class StateEntry:
    """Undo/redo for any snapshot state via a caller-supplied applier.

    *apply* receives a snapshot (old or new) and must restore it, returning
    True when it was applied.  Snapshots should be cheap copies of only
    what the edit touched; validation lives in the applier because only it
    knows whether a snapshot still matches the displayed mesh.
    """

    def __init__(self, label, apply, old_state, new_state):
        self.label = label
        self._apply = apply
        self.old_state = old_state
        self.new_state = new_state

    def undo(self) -> bool:
        return bool(self._apply(self.old_state))

    def redo(self) -> bool:
        return bool(self._apply(self.new_state))


class VertexEditEntry:
    """Positions/normals of the vertices one edit touched (undo + redo)."""

    def __init__(self, label, actor, vertex_ids, old_positions, new_positions,
                 old_normals=None, new_normals=None):
        self.label = label
        self.actor = actor
        self.vertex_ids = np.asarray(vertex_ids, dtype=np.int64)
        self.old_positions = np.asarray(old_positions, dtype=np.float32)
        self.new_positions = np.asarray(new_positions, dtype=np.float32)
        self.old_normals = (None if old_normals is None
                            else np.asarray(old_normals, dtype=np.float32))
        self.new_normals = (None if new_normals is None
                            else np.asarray(new_normals, dtype=np.float32))

    def _apply(self, positions, normals) -> bool:
        polydata = self._polydata()
        if polydata is None:
            return False
        points = vtk_to_numpy(polydata.GetPoints().GetData())
        if len(points) <= int(self.vertex_ids.max(initial=-1)):
            return False                     # mesh replaced meanwhile
        points[self.vertex_ids] = positions
        data = polydata.GetPoints().GetData()
        data.Modified()
        polydata.GetPoints().Modified()
        polydata.Modified()
        arr = polydata.GetPointData().GetNormals()
        if normals is not None and arr is not None:
            view = vtk_to_numpy(arr)
            if len(view) > int(self.vertex_ids.max()):
                view[self.vertex_ids] = normals
                arr.Modified()
        return True

    def _polydata(self):
        actor = self.actor() if callable(self.actor) else self.actor
        if actor is None:
            return None
        mapper = actor.GetMapper()
        return mapper.GetInput() if mapper is not None else None

    def undo(self) -> bool:
        return self._apply(self.old_positions, self.old_normals)

    def redo(self) -> bool:
        return self._apply(self.new_positions, self.new_normals)


class HistoryStack:
    """LIFO undo stack with a redo tail."""

    def __init__(self, limit: int = _DEFAULT_LIMIT):
        self._limit = int(limit)
        self._undo = []
        self._redo = []

    def push(self, entry):
        """Register a freshly applied edit (clears the redo tail)."""
        self._undo.append(entry)
        if len(self._undo) > self._limit:
            self._undo.pop(0)
        self._redo.clear()

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self):
        """Undo the newest entry; returns it, or None when empty."""
        if not self._undo:
            return None
        entry = self._undo.pop()
        if entry.undo():
            self._redo.append(entry)
            return entry
        return None                  # apply failed: state unchanged

    def redo(self):
        """Redo the most recently undone entry; returns it or None."""
        if not self._redo:
            return None
        entry = self._redo.pop()
        if entry.redo():
            self._undo.append(entry)
            return entry
        return None

    def peek(self):
        """The entry undo() would revert, or None."""
        return self._undo[-1] if self._undo else None

    def clear(self):
        self._undo.clear()
        self._redo.clear()
