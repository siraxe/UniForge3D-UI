"""Polygroup colour slots, per-face group assignment and visibility.

Group 0 is the implicit default polygroup: every face starts there and stays
uncoloured (no overlay) until it is assigned.  Groups 1..25 each own one entry
of :data:`POLYGROUP_COLORS`; the first still-free slot is used when the user
assigns the current lasso selection to a new polygroup (Ctrl+W in the viewer).

The controller also owns the list of *visible* groups.  ``None`` means every
group is visible; otherwise only faces whose group id is in the set are drawn.
Solo mode is simply ``{group}``.  Because VTK cannot hide individual cells of
a vtkPolyData, filtering swaps the source actors for lightweight clones that
share the source points/materials and contain only the visible polygons.
Source cell ids are preserved everywhere (selection, picking, assignment).

Kept separate from lasso_select.py because this is persistent model state,
not a transient screen-space selection.
"""

from __future__ import annotations

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from .lasso_select import (
    FaceSelection,
    _poly_cells,
    build_cell_array,
    build_selection_actors,
)


# Painted faces sit just above the model but below the green lasso highlight
# (which uses a -1.0 offset in lasso_select.build_selection_actors).
_OVERLAY_OFFSET = (-0.5, -0.5)

DEFAULT_POLYGROUP = 0

# 25 evenly spread hues cycling through five saturation/value bands, so every
# entry is clearly coloured: saturation >= 0.65, value >= 0.82, no neutral
# grays, black or white.  The list is the allocation order and walks the hue
# circle with a 12-slot (172.8 degree) step, so consecutive groups are almost
# complementary instead of blending into each other.
POLYGROUP_COLORS = (
    (1.000, 0.000, 0.000),   # red
    (0.082, 0.820, 0.731),   # teal
    (0.860, 0.129, 0.304),   # rose
    (0.198, 0.900, 0.647),   # aquamarine
    (0.960, 0.336, 0.636),   # pink
    (0.000, 1.000, 0.400),   # spring green
    (0.820, 0.082, 0.613),   # magenta
    (0.129, 0.860, 0.246),   # green
    (0.900, 0.198, 0.872),   # orchid
    (0.386, 0.960, 0.336),   # light green
    (0.800, 0.000, 1.000),   # purple
    (0.318, 0.820, 0.082),   # grass green
    (0.538, 0.129, 0.860),   # violet
    (0.591, 0.900, 0.198),   # lime
    (0.536, 0.336, 0.960),   # lavender
    (0.800, 1.000, 0.000),   # chartreuse
    (0.141, 0.082, 0.820),   # deep blue
    (0.860, 0.831, 0.129),   # olive yellow
    (0.198, 0.310, 0.900),   # royal blue
    (0.960, 0.785, 0.336),   # sand
    (0.000, 0.400, 1.000),   # blue
    (0.820, 0.436, 0.082),   # orange
    (0.129, 0.597, 0.860),   # steel blue
    (0.900, 0.366, 0.198),   # burnt orange
    (0.336, 0.885, 0.960),   # light cyan
)

MAX_POLYGROUPS = len(POLYGROUP_COLORS)
_ALL_GROUPS = frozenset(range(DEFAULT_POLYGROUP, MAX_POLYGROUPS + 1))


def group_color(group: int):
    """RGB for a 1-based polygroup id; None for the uncoloured default."""
    if 1 <= group <= MAX_POLYGROUPS:
        return POLYGROUP_COLORS[group - 1]
    return None


def _triangulated_polys(polydata):
    """(triangles, owner_polygon_ids) for the polygons of *polydata*.

    GLB files store triangles, but OBJ/PLY can contain quads or larger
    polygons; those are fan-triangulated so Split-and-save can export them.
    """
    cells = _poly_cells(polydata)
    if cells is None:
        return None, None
    offsets, connectivity = cells
    sizes = np.diff(offsets)
    if sizes.size == 0:
        return None, None
    if np.all(sizes == 3):
        return (connectivity.reshape(-1, 3),
                np.arange(sizes.size, dtype=np.int64))
    if np.all(sizes == 4):
        quads = connectivity.reshape(-1, 4)
        triangles = np.concatenate((quads[:, [0, 1, 2]],
                                    quads[:, [0, 2, 3]]), axis=0)
        return triangles, np.repeat(np.arange(sizes.size, dtype=np.int64), 2)
    triangles = []
    owners = []
    for cell_id, (start, size) in enumerate(zip(offsets[:-1], sizes)):
        if size < 3:
            continue
        ids = connectivity[start:start + size]
        for k in range(1, size - 1):
            triangles.append((ids[0], ids[k], ids[k + 1]))
            owners.append(cell_id)
    if not triangles:
        return None, None
    return (np.asarray(triangles, dtype=np.int64),
            np.asarray(owners, dtype=np.int64))


def _roots(parent: np.ndarray) -> np.ndarray:
    """Pointer-doubling until every entry of *parent* points at its root."""
    while True:
        grand = parent[parent]
        if np.array_equal(grand, parent):
            return parent
        parent = grand


def face_islands(actor, cell_ids) -> list:
    """Split *cell_ids* of *actor* into edge-connected islands.

    Two polygons belong to the same island when they share an edge (two
    consecutive corner points). Unselected faces never bridge two islands,
    so running this on the currently visible faces finds exactly the
    separate pieces the user can see. Returns a list of numpy arrays of
    cell ids, largest island first.
    """
    mapper = actor.GetMapper()
    polydata = mapper.GetInput() if mapper is not None else None
    cells = _poly_cells(polydata)
    if cells is None:
        return []
    offsets, connectivity = cells
    sizes = np.diff(offsets)
    if sizes.size == 0:
        return []
    ids = np.unique(np.asarray(cell_ids, dtype=np.int64))
    ids = ids[(ids >= 0) & (ids < sizes.size)]
    if ids.size == 0:
        return []

    # Corner-point rings of every selected polygon, flattened.
    starts = offsets[:-1][ids]
    n = sizes[ids]
    face_start = np.cumsum(n) - n
    total = int(n.sum())
    ring = np.arange(total)
    slot = np.repeat(np.arange(ids.size), n)
    pos = ring - np.repeat(face_start, n)
    base = starts[slot]
    size_rep = n[slot]
    p1 = connectivity[base + pos]
    p2 = connectivity[base + (pos + 1) % size_rep]
    edges = np.sort(np.stack((p1, p2)), axis=0)
    keys = (edges[0].astype(np.int64)
            * int(polydata.GetNumberOfPoints()) + edges[1])

    # Group duplicate keys: every edge key shared by >= 2 selected faces is
    # an internal connection. Star-link each such group to its first face.
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    sorted_slots = slot[order]
    first = np.empty(sorted_keys.size, dtype=bool)
    first[0] = True
    first[1:] = sorted_keys[1:] != sorted_keys[:-1]
    group_start = np.flatnonzero(first)
    group_size = np.diff(np.append(group_start, sorted_keys.size))
    entry_group = np.cumsum(first) - 1

    other = ~first
    if other.any():
        shared_group = np.flatnonzero(group_size > 1)
        keep = np.isin(entry_group[other], shared_group)
        a = sorted_slots[np.repeat(group_start, group_size)[other]][keep]
        b = sorted_slots[other][keep]
    else:
        a = b = np.zeros(0, dtype=np.int64)

    # Union-find (Shiloach-Vishkin style, fully vectorized).
    label = np.arange(ids.size, dtype=np.int64)
    if a.size:
        while True:
            root = _roots(label)
            ra, rb = root[a], root[b]
            lo = np.minimum(ra, rb)
            hi = np.maximum(ra, rb)
            todo = lo != hi
            if not todo.any():
                break
            np.minimum.at(label, hi[todo], lo[todo])
            label = _roots(label)
    label = _roots(label)

    # Largest island first for a predictable colour allocation order.
    root_sizes = np.bincount(label)
    roots = np.unique(label)
    islands = []
    for r in roots[np.argsort(-root_sizes[roots])]:
        islands.append(np.sort(ids[label == r]))
    return islands


def _world_points(actor, points):
    """Copy of *points* with the actor's transform baked in (float64)."""
    data = points.GetData()
    if data is None:
        return None
    vertices = vtk_to_numpy(data).astype(np.float64, copy=False)
    matrix = actor.GetMatrix()
    if matrix is None:
        return np.ascontiguousarray(vertices)
    transform = np.array(
        [[matrix.GetElement(row, col) for col in range(4)]
         for row in range(4)], dtype=np.float64)
    if np.allclose(transform, np.eye(4)):
        return np.ascontiguousarray(vertices)
    return vertices @ transform[:3, :3].T + transform[:3, 3]


def split_geometry(actors, cell_groups) -> dict:
    """World-space triangles per polygroup for Split-and-save.

    *cell_groups* maps each actor to its per-polygon group ids (or None for
    an untouched actor).  Actors sharing one vtkPolyData are exported once;
    actor/user transforms are baked into the vertices.  Group 0 contains the
    unpainted faces.  Returns ``{group_id: (vertices, triangles)}``.
    """
    parts = {}
    seen = set()
    for actor in actors:
        mapper = actor.GetMapper() if hasattr(actor, "GetMapper") else None
        polydata = mapper.GetInput() if mapper is not None else None
        if polydata is None or id(polydata) in seen:
            continue
        seen.add(id(polydata))
        points = polydata.GetPoints()
        triangles, owners = _triangulated_polys(polydata)
        if points is None or triangles is None or not owners.size:
            continue
        world = _world_points(actor, points)
        if world is None:
            continue
        cells = cell_groups.get(actor)
        needed = int(owners.max()) + 1
        if cells is None or len(cells) < needed:
            cells = np.zeros(needed, dtype=np.int64)
        owner_groups = cells[owners]
        for group in np.unique(owner_groups):
            group_tris = triangles[owner_groups == group]
            used, inverse = np.unique(group_tris, return_inverse=True)
            parts.setdefault(int(group), []).append(
                (world[used], inverse.reshape(-1, 3)))
    merged = {}
    for group, chunks in parts.items():
        offset = 0
        vertices = []
        faces = []
        for chunk_vertices, chunk_faces in chunks:
            vertices.append(chunk_vertices)
            faces.append(chunk_faces + offset)
            offset += len(chunk_vertices)
        merged[group] = (np.concatenate(vertices),
                         np.concatenate(faces))
    return merged


def visible_geometry(actors, cell_groups, visible_groups) -> tuple:
    """Merged geometry of the polygons whose group is in *visible_groups*.

    Mirrors :func:`split_geometry` but keeps every polygon of every actor
    in one mesh instead of splitting one part per group, so Delete-hidden
    can hand the whole visible remainder to a mesh tool.  *cell_groups*
    maps each actor to its per-polygon group ids (or None for an
    untouched actor); actor/user transforms are baked into the vertices.
    Polygons are fan-triangulated like in :func:`split_geometry`.

    Returns ``(vertices, triangles, keep_mask, kept, total, tri_groups)``
    where *keep_mask* marks the triangles of visible polygons, *kept* is
    the number of visible polygons, *total* the polygon count before the
    filter and *tri_groups* the per-triangle polygroup id (0 = untouched),
    aligned with the triangle rows so Delete-hidden can re-apply the
    surviving faces' groups after the delete.
    """
    visible = (None if visible_groups is None
               else frozenset(int(group) for group in visible_groups))
    parts = []                       # (vertices, faces, keep_mask) chunks
    group_parts = []                 # per-triangle polygroup ids
    kept = total = 0
    seen = set()
    for actor in actors:
        mapper = actor.GetMapper() if hasattr(actor, "GetMapper") else None
        polydata = mapper.GetInput() if mapper is not None else None
        if polydata is None or id(polydata) in seen:
            continue
        seen.add(id(polydata))
        points = polydata.GetPoints()
        triangles, owners = _triangulated_polys(polydata)
        if points is None or triangles is None or not owners.size:
            continue
        world = _world_points(actor, points)
        if world is None:
            continue
        cells = cell_groups.get(actor)
        needed = int(owners.max()) + 1
        if cells is None or len(cells) < needed:
            cells = np.zeros(needed, dtype=np.int64)
        owner_groups = cells[owners]
        total += int(np.unique(owners).size)
        if visible is None:
            tri_mask = np.ones(len(triangles), dtype=bool)
        else:
            tri_mask = np.isin(owner_groups, list(visible))
        if not tri_mask.any():
            continue
        kept += int(np.unique(owners[tri_mask]).size)
        # keep every triangle here; the delete tool drops the hidden ones
        used, inverse = np.unique(triangles, return_inverse=True)
        parts.append((world[used], inverse.reshape(-1, 3), tri_mask))
        group_parts.append(owner_groups)

    if not parts:
        return (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64),
                np.zeros(0, dtype=bool), kept, total,
                np.zeros(0, dtype=np.int64))
    vertices = []
    faces = []
    masks = []
    offset = 0
    for chunk_vertices, chunk_faces, chunk_mask in parts:
        vertices.append(chunk_vertices)
        faces.append(chunk_faces + offset)
        masks.append(chunk_mask)
        offset += len(chunk_vertices)
    return (np.concatenate(vertices), np.concatenate(faces),
            np.concatenate(masks), kept, total,
            np.concatenate(group_parts))


class PolygroupManager:
    """Lazily-allocated per-actor arrays of face-to-polygroup ids.

    ``_cells[actor]`` is a uint8 array with one entry per polygon of that
    actor's mapper input (0 = default group).  Only actors that have been
    painted at least once get an array, so untouched models cost nothing.
    """

    def __init__(self):
        self._cells = {}
        self._used = set()

    def reset(self):
        """Forget every assignment; all 25 colours become available again."""
        self._cells.clear()
        self._used.clear()

    @property
    def used_groups(self) -> tuple:
        return tuple(sorted(self._used))

    @property
    def used_count(self) -> int:
        return len(self._used)

    def _assignment(self, actor):
        cells = self._cells.get(actor)
        if cells is not None:
            return cells
        mapper = actor.GetMapper()
        polydata = mapper.GetInput() if mapper is not None else None
        if polydata is None:
            return None
        polys = _poly_cells(polydata)
        if polys is None or len(polys[0]) < 2:
            return None
        cells = np.zeros(len(polys[0]) - 1, dtype=np.uint8)
        self._cells[actor] = cells
        return cells

    def cell_groups(self, actor):
        """Group id per polygon of *actor* (0 for untouched faces).

        Unlike :meth:`_assignment` this never stores anything, so visibility
        queries for untouched models stay read-only.
        """
        cells = self._cells.get(actor)
        if cells is not None:
            return cells
        mapper = actor.GetMapper() if hasattr(actor, "GetMapper") else None
        polydata = mapper.GetInput() if mapper is not None else None
        if polydata is None:
            return None
        polys = _poly_cells(polydata)
        if polys is None or len(polys[0]) < 2:
            return None
        return np.zeros(len(polys[0]) - 1, dtype=np.uint8)

    def group_at(self, actor, cell_id: int):
        """Group id of one polygon, or None when the cell does not exist."""
        cells = self.cell_groups(actor)
        if cells is None or cell_id < 0 or cell_id >= len(cells):
            return None
        return int(cells[cell_id])

    def _prune(self):
        """Recompute used colours after writes; drop empty actors/groups."""
        used = set()
        for cells in self._cells.values():
            used.update(int(group) for group in np.unique(cells) if group)
        self._used = used
        if not used:
            self._cells.clear()
            return
        for actor, cells in list(self._cells.items()):
            if not cells.any():
                del self._cells[actor]

    def assign(self, selections):
        """Put the selected faces in the first free polygroup.

        *selections* is an iterable of ``FaceSelection`` entries.  Returns the
        new 1-based group id, or None when there is nothing to assign or all
        25 colours are already in use.  Groups emptied by the reassignment
        release their colour immediately; at capacity, such a released colour
        is reused for the new group.
        """
        by_actor = {}
        for selection in selections:
            ids = np.asarray(selection.cell_ids, dtype=np.int64)
            if ids.size == 0:
                continue
            previous = by_actor.get(selection.actor)
            by_actor[selection.actor] = (ids if previous is None
                                         else np.union1d(previous, ids))
        if not by_actor:
            return None

        entries = []
        for actor, ids in by_actor.items():
            cells = self._assignment(actor)
            if cells is None or ids.max(initial=-1) >= len(cells):
                continue
            entries.append((cells, ids))
        if not entries:
            return None

        # First choice: a colour that is not in use right now, so repainting
        # a whole group still visibly gets a new one.
        group = next((g for g in range(1, MAX_POLYGROUPS + 1)
                      if g not in self._used), None)
        if group is None:
            # At capacity, allow a colour freed by this very assignment.
            counts = np.zeros(MAX_POLYGROUPS + 1, dtype=np.int64)
            for cells in self._cells.values():
                counts += np.bincount(cells, minlength=MAX_POLYGROUPS + 1)
            for cells, ids in entries:
                counts -= np.bincount(cells[ids],
                                      minlength=MAX_POLYGROUPS + 1)
            released = {g for g in self._used if counts[g] == 0}
            group = next((g for g in range(1, MAX_POLYGROUPS + 1)
                          if g in released), None)
        if group is None:
            self._prune()        # discard empty arrays made above
            return None

        for cells, ids in entries:
            cells[ids] = group
        self._used.add(group)
        self._prune()
        return group

    def assign_multi(self, plan) -> bool:
        """Paint several face sets at once, each with a caller-chosen group.

        *plan* is an iterable of ``(actor, cell_ids, group)`` triples; the
        caller is responsible for picking free, distinct groups (see
        :meth:`PolygroupController.assign_islands`).  Returns True when at
        least one triple was applied.
        """
        applied = False
        for actor, ids, group in plan:
            cells = self._assignment(actor)
            ids = np.asarray(ids, dtype=np.int64)
            if (cells is None or ids.size == 0
                    or ids.max(initial=-1) >= len(cells)):
                continue
            cells[ids] = int(group)
            self._used.add(int(group))
            applied = True
        if applied:
            self._prune()
        return applied

    def assignments(self):
        """Yield ``(actor, group, cell_ids)`` for each non-empty face group."""
        for actor, cells in self._cells.items():
            ids = np.flatnonzero(cells)
            if ids.size == 0:
                continue
            groups = cells[ids]
            for group in np.unique(groups):
                if group:
                    yield actor, int(group), ids[groups == group]

    def snapshot(self) -> dict:
        """Copies of every painted actor's per-face group array.

        The dict is keyed by actor, exactly like the live assignments, so
        :meth:`restore_snapshot` can write it back; callers pass it through
        :meth:`PolygroupController.apply_snapshot`, which drops actors the
        mesh no longer has.
        """
        return {actor: cells.copy()
                for actor, cells in self._cells.items()}

    def restore_snapshot(self, state, known=None) -> bool:
        """Write a :meth:`snapshot` back into the assignments.

        *known* is the caller-validated set of live actors (the
        controller passes its provider's actor set); entries outside it
        are dropped.  Without *known* only actors this manager already
        holds arrays for are touched, so a stray snapshot can never
        resurrect assignments for actors the mesh dropped.  Returns True
        when at least one actor was restored.
        """
        if not state:
            return False
        if known is not None:
            state = {actor: cells for actor, cells in state.items()
                     if actor in known}
        changed = False
        for actor, cells in state.items():
            if known is None and actor not in self._cells:
                continue
            current = self._assignment(actor)
            if current is None or len(current) != len(cells):
                continue
            current[:] = cells
            changed = True
        if changed:
            self._prune()
        return changed

    def reset_actor(self, actor) -> bool:
        """Return one actor to the unpainted default group (undo support).

        True when the actor had an assignment that was dropped.
        """
        if self._cells.pop(actor, None) is None:
            return False
        self._prune()
        return True

    def forget_actors(self, actors) -> bool:
        """Drop assignments for actors that are about to be removed."""
        changed = False
        for actor in actors:
            if self._cells.pop(actor, None) is not None:
                changed = True
        if changed:
            self._prune()
        return changed


class PolygroupController:
    """Renderer-facing half of the polygroup system.

    Owns the coloured overlay actors, the visible-group filter (solo mode)
    and the filtered clone actors.  The viewer only has to call
    :meth:`assign` for Ctrl+W, the gestures (:meth:`solo`, :meth:`show_all`,
    :meth:`show_others`, :meth:`group_at`) and the lifecycle hooks (clear,
    preset visibility, actor removal, backface culling).
    """

    def __init__(self, renderer, backface_cull: bool = True,
                 on_changed=None, on_visibility_changed=None,
                 actors_provider=None):
        self._renderer = renderer
        self._manager = PolygroupManager()
        self._actors = []                    # coloured overlay actors
        self._sources = []                   # source actor per overlay
        self._backface_cull = bool(backface_cull)
        self._on_changed = on_changed
        self._on_visibility_changed = on_visibility_changed
        self._actors_provider = actors_provider

        # None = every group visible; otherwise a frozenset of group ids.
        self._visible = None
        self._clones = {}                    # id(source) -> (source, clone, ids)
        self._active_ids = set()             # ids of sources in current preset

        self._picker = vtk.vtkCellPicker()

    @property
    def capacity(self) -> int:
        return MAX_POLYGROUPS

    @property
    def used_count(self) -> int:
        return self._manager.used_count

    @property
    def used_groups(self) -> tuple:
        return self._manager.used_groups

    @property
    def actors(self) -> tuple:
        """Coloured overlay actors currently in the renderer."""
        return tuple(self._actors)

    @property
    def clone_actors(self) -> tuple:
        """Filtered stand-ins currently replacing hidden source geometry."""
        return tuple(entry[1] for entry in self._clones.values())

    @property
    def is_filtered(self) -> bool:
        """True when some polygroups are hidden."""
        return self._visible is not None

    @property
    def visible_groups(self):
        """Sorted visible group ids, or None when everything is visible."""
        return None if self._visible is None else tuple(sorted(self._visible))

    def is_group_visible(self, group: int) -> bool:
        return self._visible is None or int(group) in self._visible

    def cell_groups(self, actor):
        """Group id per polygon of *actor* (read-only when untouched)."""
        return self._manager.cell_groups(actor)

    def face_groups(self, actors=None) -> tuple:
        """Sorted group ids with at least one face in *actors*.

        Defaults to the actors of the active preset.  Group 0 means
        unpainted faces, so an untouched model reports ``(0,)``.
        """
        if actors is None:
            actors = self._actors_provider() if self._actors_provider else ()
        groups = set()
        for actor in actors:
            cells = self._manager.cell_groups(actor)
            if cells is None or not len(cells):
                continue
            groups.update(int(group) for group in np.unique(cells))
        return tuple(sorted(groups))

    # -- visibility list ----------------------------------------------------
    def show_all(self) -> bool:
        """Make every polygroup visible again (leave solo mode)."""
        return self.set_visible_groups(None)

    def solo(self, group: int) -> bool:
        """Show only *group* (the default group 0 counts as a group)."""
        group = int(group)
        if not DEFAULT_POLYGROUP <= group <= MAX_POLYGROUPS:
            return False
        return self.set_visible_groups((group,))

    def show_others(self, group: int) -> bool:
        """Hide *group* and show every other group."""
        group = int(group)
        if not DEFAULT_POLYGROUP <= group <= MAX_POLYGROUPS:
            return False
        return self.set_visible_groups(_ALL_GROUPS - {group})

    def hide_group(self, group: int) -> bool:
        """Add *group* to the hidden set, keeping the other visible groups.

        Only meaningful while a filter is active; the last visible group is
        kept on screen so the mesh never disappears completely.
        """
        group = int(group)
        if self._visible is None or len(self._visible) <= 1:
            return False
        if group not in self._visible:
            return False
        return self.set_visible_groups(self._visible - {group})

    def invert_visibility(self) -> bool:
        """Show exactly the groups that are hidden right now.

        With one soloed group this reveals every other group; with two or
        more visible groups it swaps the whole visible/hidden sets.
        """
        if self._visible is None:
            return False
        return self.set_visible_groups(_ALL_GROUPS - self._visible)

    def set_visible_groups(self, groups) -> bool:
        """Replace the visible-group list (None = show all groups)."""
        if groups is None:
            normalized = None
        else:
            normalized = frozenset(
                int(group) for group in groups
                if DEFAULT_POLYGROUP <= int(group) <= MAX_POLYGROUPS)
            if normalized >= _ALL_GROUPS:
                normalized = None
        if normalized == self._visible:
            return False
        self._visible = normalized
        self._rebuild()
        self._notify_visibility()
        return True

    def visible_cell_ids(self, actor):
        """Original cell ids of *actor* that the filter currently shows."""
        cells = self._manager.cell_groups(actor)
        if cells is None:
            return None
        if self._visible is None:
            return np.arange(len(cells), dtype=np.int64)
        return np.flatnonzero(self._group_mask()[cells])

    def filter_cell_ids(self, actor, cell_ids):
        """Keep only the ids in *cell_ids* whose group is visible."""
        cell_ids = np.asarray(cell_ids, dtype=np.int64)
        if self._visible is None or cell_ids.size == 0:
            return cell_ids
        cells = self._manager.cell_groups(actor)
        if cells is None:
            return np.zeros(0, dtype=np.int64)
        usable = (cell_ids >= 0) & (cell_ids < len(cells))
        kept = cell_ids[usable]
        if kept.size == 0:
            return kept
        mask = self._group_mask()[cells[kept]]
        return kept[mask]

    # -- picking ------------------------------------------------------------
    def group_at(self, x: float, y: float):
        """Group id at VTK display coordinates, or None over empty space.

        Works with both regular source actors and the filtered clone actors;
        clone cell indices are translated back to source cell ids.
        """
        self._picker.Pick(int(round(x)), int(round(y)), 0.0, self._renderer)
        prop = self._picker.GetViewProp()
        cell_id = int(self._picker.GetCellId())
        if prop is None or cell_id < 0 or not prop.GetVisibility():
            return None
        if getattr(prop, "_polygroup_clone", False):
            source = getattr(prop, "_polygroup_source", None)
            cell_map = getattr(prop, "_polygroup_cell_map", None)
            if source is None or cell_map is None or cell_id >= len(cell_map):
                return None
            cell_id = int(cell_map[cell_id])
        else:
            source = prop
        return self._manager.group_at(source, cell_id)

    # -- lifecycle ----------------------------------------------------------
    def clear(self):
        """Remove overlays/clones and return every face to the default group."""
        for source_id in list(self._clones):
            self._remove_clone(source_id)
        for actor in self._actors:
            self._renderer.RemoveActor(actor)
        self._actors = []
        self._sources = []
        self._manager.reset()
        self._visible = None
        self._active_ids = set()

    def assign(self, selections, reveal: bool = False):
        """Paint *selections* with the first free polygroup colour.

        With *reveal* the new group is added to the visible set when a
        filter is active, so freshly painted faces do not vanish.  Returns
        the 1-based group id, or None when there is nothing to assign / all
        colours are in use.
        """
        group = self._manager.assign(selections)
        if group is None:
            return None
        if reveal and self._visible is not None:
            self._visible = self._visible | {group}
        self._rebuild()
        self._notify()
        return group

    def assign_islands(self, selections, reveal: bool = False) -> tuple:
        """Give the largest edge-connected islands their own colour.

        *selections* is an iterable of ``FaceSelection`` entries; islands
        never span actors and never bridge across unselected (or hidden)
        faces.  Islands are painted largest-first into the free colours;
        when there are more islands than free colours the smaller ones stay
        unpainted.  Returns ``(new_group_ids, island_count)``.
        """
        islands = []
        for selection in selections:
            ids = np.asarray(selection.cell_ids, dtype=np.int64)
            if ids.size == 0:
                continue
            for part in face_islands(selection.actor, ids):
                islands.append((selection.actor, part))
        if not islands:
            return [], 0
        islands.sort(key=lambda item: len(item[1]), reverse=True)
        used = set(self._manager.used_groups)
        free = [g for g in range(1, MAX_POLYGROUPS + 1) if g not in used]
        chosen = islands[:len(free)]
        if not chosen:
            return [], len(islands)
        plan = [(actor, ids, group)
                for (actor, ids), group in zip(chosen, free)]
        if not self._manager.assign_multi(plan):
            return [], len(islands)
        applied = [group for _, _, group in plan]
        if reveal and self._visible is not None:
            self._visible = self._visible | set(applied)
        self._rebuild()
        self._notify()
        return applied, len(islands)

    def snapshot(self) -> dict:
        """Per-actor copies of the face-group arrays (undo/redo support).

        Feed the result back through :meth:`apply_snapshot`; never modify
        it in place, the arrays are private copies but the actor keys are
        the live actors.
        """
        return self._manager.snapshot()

    def apply_snapshot(self, state) -> bool:
        """Restore a :meth:`snapshot`; True when anything changed.

        Snapshots die with a model reload (the viewer clears its history
        then, so this is only a safety net): entries whose actor is no
        longer in the current actor set are dropped, and live actors the
        snapshot does not mention were unpainted at snapshot time, so they
        are reset to the default group.  On success the overlays and the
        visibility filter are rebuilt, like after an assign.
        """
        known = (self._actors_provider() if self._actors_provider
                 else self._sources)
        live = {actor: cells for actor, cells in (state or {}).items()
                if actor in set(known)}
        changed = self._manager.restore_snapshot(live, known=set(known))
        restored = set(live)
        for actor in known:
            if actor not in restored and self._manager.reset_actor(actor):
                changed = True
        if not changed:
            return False
        self._rebuild()
        self._notify()
        self._notify_visibility()
        return True

    def restore(self, actor, groups) -> bool:
        """Re-apply per-polygon group ids to a freshly loaded mesh.

        Used by Delete-hidden: the surviving faces come back in their
        original order, so *groups* (one entry per polygon of *actor*'s
        mesh) restores every polygroup that still has faces, with its
        original colour.  Groups that lost all their faces simply
        disappear; untouched polygons (id 0) stay unassigned.  Returns
        True when at least one group came back.
        """
        groups = np.asarray(groups, dtype=np.int64).ravel()
        plan = []
        for group in np.unique(groups):
            if not group:
                continue
            plan.append((actor, np.flatnonzero(groups == group), int(group)))
        if not plan or not self._manager.assign_multi(plan):
            return False
        if self._visible is not None:
            self._visible = self._visible | {int(g) for _, _, g in plan}
        self._rebuild()
        self._notify()
        self._notify_visibility()
        return True

    def reset_to_default(self) -> bool:
        """Put every face back into group 0 (the uncoloured default).

        Used by Ctrl+W when nothing is selected and the whole mesh is
        visible: it clears all painted groups and their colours.
        """
        if not self._manager.used_count:
            return False
        self._manager.reset()
        self._rebuild()
        self._notify()
        if self.is_filtered:
            self._notify_visibility()
        return True

    def forget_actors(self, actors) -> bool:
        """Drop assignments and clones belonging to actors being removed."""
        changed = self._manager.forget_actors(actors)
        # Refresh after the caller has already dropped the actors from its
        # own lists, so the provider no longer reports them as active.
        self._rebuild()
        self._notify()
        self._notify_visibility()
        return changed

    def set_backface_culling(self, enabled: bool):
        self._backface_cull = bool(enabled)
        for actor in self._actors:
            prop = actor.GetProperty()
            if self._backface_cull:
                prop.BackfaceCullingOn()
            else:
                prop.BackfaceCullingOff()
        # Clone actors share their source property, so the viewer's normal
        # model/texture pass already updated them.

    def sync_visibility(self):
        """Re-apply the preset's active actors and the group filter.

        Called after preset changes: source actors may have swapped between
        the model and textured variant, which requires rebuilding the
        filtered clones and re-hiding the matching colour overlays.
        """
        self.refresh()

    def refresh(self):
        """Rebuild clones after the callers' active actor set changed."""
        self._sync_clones()
        self._sync_overlay_visibility()

    # -- rendering internals ------------------------------------------------
    def _group_mask(self):
        mask = np.zeros(256, dtype=bool)
        for group in self._visible:
            mask[group] = True
        return mask

    def _rebuild(self):
        self._sync_clones()
        for actor in self._actors:
            self._renderer.RemoveActor(actor)
        self._actors = []
        self._sources = []
        for source, group, cell_ids in self._manager.assignments():
            if not self.is_group_visible(group):
                continue
            color = group_color(group)
            if color is None:
                continue
            overlays = build_selection_actors(
                [FaceSelection(source, cell_ids)], self._backface_cull,
                color=color, polygon_offset=_OVERLAY_OFFSET)
            if not overlays:
                continue
            overlay = overlays[0]
            overlay._polygroup_source = source
            overlay._polygroup_id = group
            self._renderer.AddActor(overlay)
            self._actors.append(overlay)
            self._sources.append(source)
        self._sync_overlay_visibility()

    def _sync_overlay_visibility(self):
        for actor, source in zip(self._actors, self._sources):
            group = getattr(actor, "_polygroup_id", DEFAULT_POLYGROUP)
            actor.SetVisibility(id(source) in self._active_ids
                                and self.is_group_visible(group))

    def _sync_clones(self):
        sources = []
        if self._actors_provider is not None:
            sources = [actor for actor in self._actors_provider()
                       if not getattr(actor, "_polygroup_clone", False)]
        active = {id(actor) for actor in sources}
        self._active_ids = active
        for source_id in list(self._clones):
            if source_id not in active:
                self._remove_clone(source_id)
        for source in sources:
            self._sync_clone_for(source)

    def _sync_clone_for(self, source):
        source_id = id(source)
        if self._visible is None:
            source.SetVisibility(True)
            self._remove_clone(source_id)
            return
        cells = self._manager.cell_groups(source)
        if cells is None:
            self._remove_clone(source_id)
            return
        cell_ids = np.flatnonzero(self._group_mask()[cells])
        if cell_ids.size == len(cells):
            source.SetVisibility(True)
            self._remove_clone(source_id)
            return
        source.SetVisibility(False)
        if cell_ids.size == 0:
            self._remove_clone(source_id)
            return
        existing = self._clones.get(source_id)
        if existing is not None:
            if np.array_equal(existing[2], cell_ids):
                existing[1].SetVisibility(True)
                return
            self._remove_clone(source_id)
        clone = self._build_clone(source, cell_ids)
        if clone is None:
            source.SetVisibility(False)
            return
        self._clones[source_id] = (source, clone, cell_ids)

    def _build_clone(self, source, cell_ids):
        mapper = source.GetMapper()
        polydata = mapper.GetInput() if mapper is not None else None
        if polydata is None:
            return None
        cell_array = build_cell_array(polydata, cell_ids)
        if cell_array is None:
            return None

        filtered = vtk.vtkPolyData()
        filtered.SetPoints(polydata.GetPoints())
        filtered.GetPointData().ShallowCopy(polydata.GetPointData())
        filtered.SetPolys(cell_array)

        clone_mapper = vtk.vtkPolyDataMapper()
        clone_mapper.SetInputData(filtered)
        clone_mapper.SetScalarVisibility(mapper.GetScalarVisibility())
        if mapper.GetScalarVisibility():
            clone_mapper.SetScalarMode(mapper.GetScalarMode())
            clone_mapper.SetColorMode(mapper.GetColorMode())
            clone_mapper.SetScalarRange(mapper.GetScalarRange())
            clone_mapper.SetLookupTable(mapper.GetLookupTable())
            clone_mapper.SetUseLookupTableScalarRange(
                mapper.GetUseLookupTableScalarRange())

        clone = vtk.vtkActor()
        clone.SetMapper(clone_mapper)
        # Share the property object instead of copying it: presets, textures,
        # opacity and backface culling then keep applying to the clone.
        clone.SetProperty(source.GetProperty())
        clone.SetOrigin(source.GetOrigin())
        clone.SetPosition(source.GetPosition())
        clone.SetOrientation(source.GetOrientation())
        clone.SetScale(source.GetScale())
        matrix = vtk.vtkMatrix4x4()
        matrix.DeepCopy(source.GetMatrix())
        clone.SetUserMatrix(matrix)
        clone.PickableOn()
        clone._polygroup_clone = True
        clone._polygroup_source = source
        clone._polygroup_cell_map = cell_ids
        self._renderer.AddActor(clone)
        return clone

    def _remove_clone(self, source_id):
        entry = self._clones.pop(source_id, None)
        if entry is not None:
            self._renderer.RemoveActor(entry[1])

    def _notify(self):
        if self._on_changed is not None:
            self._on_changed()

    def _notify_visibility(self):
        if self._on_visibility_changed is not None:
            self._on_visibility_changed()
