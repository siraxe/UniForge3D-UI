"""Localized Laplacian ('umbrella') smoothing on the displayed mesh.

Implements the core every sculpting package shares (ZBrush documents it as
"averaging the displacement of vertices with the displacements of
neighboring vertices"): per dab, every vertex inside the brush radius moves
toward the average of its one-ring neighbours, modulated by a quadratic
radial falloff and the brush strength.

Design notes:
* One dab per call; a stroke is a series of dabs spaced along the mouse
  path, which makes the result speed-independent.
* An invisible inner core (a fraction of the brush radius) multiplies the
  falloff, so the centre of the dab smooths harder than its rim.
* Fully vectorized numpy (CSR adjacency + ragged gather + add.reduceat),
  so a dab over ~50k vertices costs single-digit milliseconds even on a
  0.5M-triangle mesh.
* Vertex positions are written back IN PLACE through a writable
  vtk_to_numpy view (no array reallocation per dab).
* Vertex normals of the moved region are recomputed from the current
  incident face normals so shading follows the deformation.
* Topology caches (adjacency, vertex->cells) never go stale - smoothing
  moves vertices, never changes connectivity.  The point locator is rebuilt
  once per stroke; within a stroke its buckets may drift slightly, which is
  filtered by an exact distance re-test per dab.
"""

from __future__ import annotations

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


class MeshDeformer:
    """Applies deforming strokes (smooth dabs, grabs) to a mesh polydata."""

    def __init__(self):
        # id(polydata) -> (polydata, topology); VTK objects are unhashable,
        # and the identity guard keeps id-reuse after gc from lying to us
        self._topology = {}
        self._locator = None      # point locator, rebuilt per stroke
        self._locator_pd = None
        self._locator_snap = None  # frozen dataset the locator points at
        # per-stroke undo capture (valid between begin_stroke/end_stroke)
        self._rec_ids = []
        self._rec_pos = []
        self._rec_nrm = []
        self._captured = None
        # cells whose face normals changed this stroke (accumulated by
        # _refresh_normals) so the caller can patch ITS face-normal cache
        # regionally at end_stroke instead of rebuilding the whole mesh
        self._touched_cells = []
        # optional world-space symmetry for this stroke: a
        # tools.mesh_edit.symmetry.SymmetryMapper set by the brush
        # controller when the Edit-mesh panel's X/Y/Z checkboxes are on
        # (None = plain asymmetric editing)
        self.symmetry = None

    # -- stroke lifecycle ----------------------------------------------------
    def begin_stroke(self, actor) -> bool:
        """Prepare smoothing on *actor*; False when it has no polygons."""
        polydata = self._polydata(actor)
        if polydata is None:
            return False
        self._topology_of(polydata)
        # Fresh locator per stroke - but NOT pointed at the live polydata:
        # VTK locators silently rebuild on every query once the dataset's
        # MTime moves, and each dab calls polydata.Modified(), so a live
        # locator would pay a full O(all-points) grid rebuild PER DAB on
        # dense meshes.  Instead the locator is built on a detached
        # snapshot (deep-copied points, shared connectivity that smoothing
        # never touches) whose MTime stays frozen, so the grid is built
        # once per stroke and every radius query is O(bucket) again.
        # Buckets drift as dabs move vertices - filtered by the exact
        # distance re-test per dab, as before.
        snap = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.DeepCopy(polydata.GetPoints())
        snap.SetPoints(points)
        snap.SetPolys(polydata.GetPolys())
        locator = vtk.vtkStaticPointLocator()
        locator.SetDataSet(snap)
        locator.BuildLocator()
        self._locator = locator
        self._locator_pd = polydata
        self._locator_snap = snap
        # undo bookkeeping: first-touch snapshots for every vertex the
        # stroke ends up moving (see end_stroke)
        self._rec_ids = []
        self._rec_pos = []
        self._rec_nrm = []
        self._captured = None
        self._touched_cells = []
        return True

    def end_stroke(self):
        """Finish the stroke; returns an undo record or None.

        The record carries, for every vertex the stroke touched: its id,
        its position before the stroke started and after it ended (same
        for normals when the mesh has any) - exactly what
        ``history.VertexEditEntry`` needs.  Memory is proportional to the
        touched region, not the mesh.

        When face normals changed (i.e. the stroke moved anything), the
        record also carries ``cells`` + ``cell_normals``: the mesh-space
        ids of every face the stroke touched and their final area-weighted
        normals - letting the caller patch its hover face-normal cache
        regionally instead of rebuilding it over the whole mesh.
        """
        record = None
        polydata = self._locator_pd
        if self._rec_ids and polydata is not None:
            ids = np.concatenate(self._rec_ids)
            points = np.asarray(
                vtk_to_numpy(polydata.GetPoints().GetData()),
                dtype=np.float32)
            normals = polydata.GetPointData().GetNormals()
            record = {
                "polydata": polydata,
                "ids": ids,
                "old_positions": np.concatenate(self._rec_pos),
                "new_positions": points[ids].copy(),
                "old_normals": (np.concatenate(self._rec_nrm)
                                if self._rec_nrm else None),
                "new_normals": (
                    np.asarray(vtk_to_numpy(normals)[ids],
                               dtype=np.float32).copy()
                    if normals is not None else None),
            }
            if self._touched_cells:
                topo = self._topology_of(polydata)
                cells = np.unique(np.concatenate(self._touched_cells))
                record["cells"] = cells
                record["cell_normals"] = self._face_normals(topo, points,
                                                            cells)
        self._rec_ids = []
        self._rec_pos = []
        self._rec_nrm = []
        self._captured = None
        self._touched_cells = []
        self._locator = None
        self._locator_pd = None
        self._locator_snap = None
        return record

    # -- smoothing ------------------------------------------------------------
    def dab(self, actor, center_world, radius_world: float,
            strength: float = 0.5, inner_ratio: float = 0.0,
            inner_boost: float = 1.0,
            normal_world=None, flatten: float = 0.0,
            flatten_depth: float = 0.0, allowed_verts=None) -> int:
        """One smooth dab at *center_world*; returns moved-vertex count.

        Every vertex within *radius_world* of the centre moves toward its
        one-ring average by strength * falloff (quadratic falloff from the
        centre to the rim).  Vertices inside the invisible inner core at
        ``inner_ratio * radius_world`` get their falloff multiplied by
        *inner_boost*, so the centre of the brush digs deeper than the
        rim.  Strength above 1 adds full smoothing passes per dab instead
        of scaling a single pass past 1.0 - overshooting the one-ring
        average would make the surface oscillate.  Positions and normals
        are written in place; callers should re-render.

        *allowed_verts* optionally restricts the dab to a boolean
        per-vertex mask (the visible-polygroup filter in the viewer), so
        faces hidden by the filter stay put.  The mask is applied to the
        radius query before the falloff is built.  Vertices shared with
        hidden faces still participate as one-ring neighbours - the mesh
        is not split at polygroup borders, matching the move brush's
        existing per-group mask behaviour.

        With *normal_world* (the picked face normal) and *flatten* > 0 a
        second pass presses the region against a plane: the plane passes
        through the dab centre offset *flatten_depth* * radius_world
        *inward* along the normal, and every vertex in the brush moves
        toward that plane by ``flatten * falloff * fade`` of its signed
        distance - so bumps in front of the plane get shaved down while
        dips behind it rise, which visibly flattens the surface while it
        smooths.  The extra smootherstep *fade* feathers the effect to
        zero at the brush rim so it never leaves a terrace.  Weights stay
        capped at 1.0, so a vertex can approach the plane but never cross
        it (no oscillation).

        With ``self.symmetry`` set (the Edit-mesh panel's X/Y/Z planes)
        a second pass runs the same dab at the mirrored centre/normal -
        the mirrored camera ray's hit - so both sides deform identically
        without requiring any vertex symmetry.  Each pass damps its
        cross-plane displacement near the enabled planes (see
        ``SymmetryPlanes.fade_mult``), so strokes across the centerline
        blend instead of tearing.
        """
        polydata = self._polydata(actor)
        if (polydata is None or self._locator is None
                or self._locator_pd is not polydata):
            return 0
        topo = self._topology_of(polydata)
        moved = self._dab_pass(polydata, topo, actor, center_world,
                               radius_world, strength, inner_ratio,
                               inner_boost, normal_world, flatten,
                               flatten_depth, allowed_verts)
        sym = self.symmetry
        if sym is not None:
            # Mirrored pass: the dab reflected across the enabled world
            # planes is the hit the mirrored camera would get, so the
            # brush reforms whatever geometry sits there - no vertex
            # symmetry required (see tools.mesh_edit.symmetry).
            moved = max(moved, self._dab_pass(
                polydata, topo, actor, sym.reflect_point(center_world),
                radius_world, strength, inner_ratio, inner_boost,
                (None if normal_world is None
                 else sym.reflect_dir(normal_world)),
                flatten, flatten_depth, allowed_verts))
        return moved

    def _dab_pass(self, polydata, topo, actor, center_world,
                  radius_world, strength, inner_ratio, inner_boost,
                  normal_world, flatten, flatten_depth,
                  allowed_verts) -> int:
        """One dab pass at *center_world* (the symmetry dispatcher in
        :meth:`dab` runs it twice when planes are enabled)."""
        mesh_center, mesh_radius = self._to_mesh(actor, center_world,
                                                 radius_world)
        if mesh_center is None or mesh_radius <= 0.0:
            return 0

        ids = vtk.vtkIdList()
        self._locator.FindPointsWithinRadius(float(mesh_radius),
                                             tuple(mesh_center), ids)
        count = ids.GetNumberOfIds()
        if count == 0:
            return 0
        candidates = np.fromiter((ids.GetId(i) for i in range(count)),
                                 np.int64, count)

        # Writable view of the live point array (verified float32-safe).
        points = vtk_to_numpy(polydata.GetPoints().GetData())
        dist = np.linalg.norm(points[candidates] - mesh_center, axis=1)
        # exact re-test: the locator's buckets predate earlier dabs
        inside = dist <= mesh_radius
        if allowed_verts is not None:
            inside &= np.asarray(allowed_verts, dtype=bool)[candidates]
        verts = candidates[inside]
        if len(verts) == 0:
            return 0
        falloff = (1.0 - dist[inside] / mesh_radius) ** 2
        # the flatten pass keeps this clean quadratic: the inner-core boost
        # below is folded into *falloff* only (smoothing digs harder in the
        # middle, but flattening stays purely radial so the pressed plane
        # stays an even terrace instead of a dimple)
        flatten_falloff = falloff
        if inner_ratio > 0.0 and inner_boost != 1.0:
            core = dist[inside] <= inner_ratio * mesh_radius
            falloff = np.where(core, falloff * inner_boost, falloff)

        self._capture_before(polydata, verts)
        pre = (points[verts].copy()
               if self.symmetry is not None else None)  # falloff baseline
        # Strength 0..1 is one fractional pass (unchanged feel); above 1
        # it becomes whole passes plus the remainder, up to the popup's
        # 3.0 maximum.  This smooths harder without overshooting the
        # one-ring average (which would oscillate the surface).
        strength = min(max(float(strength), 0.0), 3.0)
        whole = int(strength)
        moved = 0
        for _ in range(whole):
            moved = max(moved, self._laplacian(topo, points, verts,
                                              falloff, 1.0))
        remainder = strength - whole
        if remainder > 0.0:
            moved = max(moved, self._laplacian(topo, points, verts,
                                               falloff, remainder))
        if flatten > 0.0 and normal_world is not None:
            n = self._to_mesh_normal(actor, normal_world)
            if n is not None:
                # plane through the centre, pushed inward along the normal
                plane_center = (mesh_center
                                - n * (float(flatten_depth) * mesh_radius))
                # signed distance to the plane, on the post-Laplacian
                # positions so both passes compose on the same dab
                d = (points[verts] - plane_center) @ n
                # smootherstep edge fade (C2-continuous from full strength
                # at the centre to zero at the rim) layered on the quadratic
                # falloff, so the flatten blends into the untouched surface
                # with no visible terrace at the brush border
                t = np.clip(dist[inside] / mesh_radius, 0.0, 1.0)
                fade = 1.0 - t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
                weight = np.minimum(
                    np.clip(flatten, 0.0, 1.0) * flatten_falloff * fade,
                    1.0)
                points[verts] = points[verts] - np.outer(d * weight, n)
                moved = max(moved, len(verts))
        if moved == 0:
            return 0
        if pre is not None:
            # Near-plane guard (SymmetryPlanes.fade_mult): damp the
            # cross-plane components of this pass's displacement as the
            # plane approaches, so the pass and its mirror image can
            # never pull the centerline apart.
            disp = points[verts] - pre
            full, trans = self._world_xform(actor)
            mult = self.symmetry.fade_mult(
                points[verts] @ full.T + trans, radius_world)
            points[verts] = pre + disp * mult
        data = polydata.GetPoints().GetData()
        data.Modified()
        polydata.GetPoints().Modified()
        polydata.Modified()
        self._refresh_normals(polydata, topo, verts)
        return moved

    def _capture_before(self, polydata, verts):
        """Snapshot first-touch vertices once per stroke (undo record)."""
        if self._captured is None:
            self._captured = np.zeros(
                polydata.GetPoints().GetNumberOfPoints(), dtype=bool)
        fresh = verts[~self._captured[verts]]
        if len(fresh) == 0:
            return
        self._rec_ids.append(fresh)
        points = vtk_to_numpy(polydata.GetPoints().GetData())
        self._rec_pos.append(points[fresh].astype(np.float32, copy=True))
        normals = polydata.GetPointData().GetNormals()
        if normals is not None:
            self._rec_nrm.append(
                vtk_to_numpy(normals)[fresh].astype(np.float32, copy=True))
        self._captured[fresh] = True

    def _laplacian(self, topo, points, verts, falloff, strength) -> int:
        """Umbrella-smooth *verts* in place; returns how many moved."""
        starts = topo["indptr"][verts]
        ends = topo["indptr"][verts + 1]
        counts = ends - starts
        keep = counts > 0                      # isolated verts: no ring
        if not keep.all():
            verts = verts[keep]
            falloff = falloff[keep]
            starts = starts[keep]
            counts = counts[keep]
        total = int(counts.sum())
        if total == 0:
            return 0
        cum = np.concatenate([[0], np.cumsum(counts)[:-1]])
        flat = (np.repeat(starts, counts)
                + np.arange(total) - np.repeat(cum, counts))
        gathered = points[topo["indices"][flat]].astype(np.float64)
        averages = np.add.reduceat(gathered, cum, axis=0) / counts[:, None]
        weight = np.minimum(np.clip(strength, 0.0, 1.0) * falloff, 1.0)
        points[verts] = (points[verts]
                         + (averages - points[verts]) * weight[:, None])
        return len(verts)

    # -- masking --------------------------------------------------------------
    def vertex_mask_for_cells(self, actor, included) -> np.ndarray | None:
        """Per-vertex mask covering every face set in boolean *included*.

        The brushes use this while the polygroup visibility filter hides
        some groups: a vertex is eligible when at least one of its incident
        faces is visible, so hidden-only geometry stays put.  Vertices on a
        group border are shared with the mesh (nothing is split), exactly
        like the move brush's per-group mask.

        Returns None when *actor* has no polydata; a size mismatch (stale
        group array) returns an all-False mask, so a broken restriction can
        never silently turn into an unrestricted edit.
        """
        polydata = self._polydata(actor)
        if polydata is None:
            return None
        topo = self._topology_of(polydata)
        included = np.asarray(included, dtype=bool)
        count = polydata.GetNumberOfPoints()
        if included.size != len(topo["sizes"]):
            return np.zeros(count, dtype=bool)
        mask = np.zeros(count, dtype=bool)
        if included.any():
            cell_of = np.repeat(np.arange(included.size), topo["sizes"])
            mask[topo["conn"][included[cell_of]]] = True
        return mask

    # -- normals --------------------------------------------------------------
    def _refresh_normals(self, polydata, topo, moved):
        """Recompute vertex normals of the moved region (area weighted).

        The point array is read as a zero-copy view; every float64
        conversion happens on region-sized gathers only (a full-mesh
        float64 copy here cost O(all points) PER DAB on dense meshes).
        """
        arr = polydata.GetPointData().GetNormals()
        if arr is None or len(moved) == 0:
            return
        points = vtk_to_numpy(polydata.GetPoints().GetData())
        starts = topo["vcell_ptr"][moved]
        ends = topo["vcell_ptr"][moved + 1]
        counts = ends - starts
        keep = counts > 0
        if not keep.all():
            moved = moved[keep]
            starts = starts[keep]
            counts = counts[keep]
            if len(moved) == 0:
                return
        total = int(counts.sum())
        cum = np.concatenate([[0], np.cumsum(counts)[:-1]])
        flat = (np.repeat(starts, counts)
                + np.arange(total) - np.repeat(cum, counts))
        cells = topo["vcell_ids"][flat]
        uniq = np.unique(cells)
        # remember which faces changed so end_stroke can hand the caller
        # a regional face-normal patch (see the record's cells/cell_normals)
        self._touched_cells.append(uniq)
        face_normals = self._face_normals(topo, points, uniq)
        slot = np.searchsorted(uniq, cells)
        sums = np.add.reduceat(face_normals[slot], cum, axis=0)
        lengths = np.linalg.norm(sums, axis=1)
        ok = lengths > 1e-12
        if not ok.any():
            return
        view = vtk_to_numpy(arr)
        view[moved[ok]] = (sums[ok] / lengths[ok, None]).astype(view.dtype)
        arr.Modified()

    def _face_normals(self, topo, points, cells):
        """Area-weighted face normals for *cells* (Newell for polygons).

        *points* may be a zero-copy float32 view of the live array; the
        float64 conversion runs on the region gather only.
        """
        conn, offs, sizes = topo["conn"], topo["offs"], topo["sizes"]
        out = np.zeros((len(cells), 3))
        tri = sizes[cells] == 3
        tri_cells = cells[tri]
        if len(tri_cells):
            s = offs[tri_cells]
            p0 = points[conn[s]].astype(np.float64)
            p1 = points[conn[s + 1]].astype(np.float64)
            p2 = points[conn[s + 2]].astype(np.float64)
            out[tri] = np.cross(p1 - p0, p2 - p0)
        for row, cell in enumerate(cells[~tri]):
            ring = points[conn[offs[cell]:offs[cell + 1]]].astype(np.float64)
            acc = np.zeros(3)
            for k in range(len(ring)):
                acc = acc + np.cross(ring[k], ring[(k + 1) % len(ring)])
            out[row] = acc
        return out

    # -- caches -----------------------------------------------------------------
    def _polydata(self, actor):
        mapper = actor.GetMapper()
        if mapper is None:
            return None
        polydata = mapper.GetInput()
        if polydata is None or polydata.GetNumberOfPolys() == 0:
            return None
        return polydata

    def _topology_of(self, polydata) -> dict:
        """One-ring adjacency (CSR) + vertex->cells (CSR), built once."""
        key = id(polydata)
        cached = self._topology.get(key)
        if cached is not None and cached[0] is polydata:
            return cached[1]
        cells = polydata.GetPolys()
        conn = np.asarray(vtk_to_numpy(cells.GetConnectivityArray()),
                          dtype=np.int64)
        offs = np.asarray(vtk_to_numpy(cells.GetOffsetsArray()),
                          dtype=np.int64)
        sizes = np.diff(offs)
        ncells = len(sizes)
        cell_of = np.repeat(np.arange(ncells), sizes)

        # directed polygon edges -> undirected -> CSR adjacency
        starts = offs[:-1]
        pos = np.arange(len(conn))
        nxt = np.where(pos + 1 < offs[cell_of + 1], pos + 1, starts[cell_of])
        edges = np.unique(np.sort(np.stack([conn, conn[nxt]], 1), 1), axis=0)
        src = np.concatenate([edges[:, 0], edges[:, 1]])
        dst = np.concatenate([edges[:, 1], edges[:, 0]])
        order = np.argsort(src, kind="stable")
        src, dst = src[order], dst[order]
        nverts = polydata.GetNumberOfPoints()
        indptr = np.concatenate(
            [[0], np.cumsum(np.bincount(src, minlength=nverts))]
        ).astype(np.int64)

        # vertex -> incident cells (each vertex appears once per polygon)
        vorder = np.argsort(conn, kind="stable")
        vcell_ptr = np.concatenate(
            [[0], np.cumsum(np.bincount(conn[vorder], minlength=nverts))]
        ).astype(np.int64)
        vcell_ids = cell_of[vorder]

        topo = {"conn": conn, "offs": offs, "sizes": sizes,
                "indptr": indptr, "indices": dst,
                "vcell_ptr": vcell_ptr, "vcell_ids": vcell_ids}
        self._topology[key] = (polydata, topo)
        return topo

    @staticmethod
    def _world_xform(actor):
        """Actor's world linear part and translation, or identity."""
        matrix = actor.GetMatrix()
        if matrix is None:
            return np.eye(3), np.zeros(3)
        m = np.array([[matrix.GetElement(r, c) for c in range(4)]
                      for r in range(4)])
        return m[:3, :3], m[:3, 3]

    @staticmethod
    def _to_mesh_normal(actor, normal_world):
        """World-space direction -> unit mesh-space direction, or None.

        Uses the inverse-transpose of the actor matrix so non-uniform
        transforms stay correct; the result is re-normalised because the
        matrix may scale.
        """
        n = np.asarray(normal_world, dtype=np.float64)
        length = float(np.linalg.norm(n))
        if length < 1e-12:
            return None
        matrix = actor.GetMatrix()
        if matrix is None:
            return n / length
        m = np.array([[matrix.GetElement(r, c) for c in range(3)]
                      for r in range(3)])
        try:
            inv = np.linalg.inv(m)
        except np.linalg.LinAlgError:
            return None
        n = inv.T @ (n / length)
        length = float(np.linalg.norm(n))
        if length < 1e-12:
            return None
        return n / length

    @staticmethod
    def _to_mesh(actor, center_world, radius_world):
        """World centre/radius -> mesh space (uniform scale assumed)."""
        matrix = actor.GetMatrix()
        if matrix is None:
            return (np.asarray(center_world, dtype=np.float64),
                    float(radius_world))
        m = np.array([[matrix.GetElement(r, c) for c in range(4)]
                      for r in range(4)])
        try:
            inv = np.linalg.inv(m)
        except np.linalg.LinAlgError:
            return None, 0.0
        center = (inv @ np.append(np.asarray(center_world,
                                              dtype=np.float64), 1.0))[:3]
        scale = float(np.linalg.norm(m[:3, 0]))
        if scale < 1e-12:
            return None, 0.0
        return center, radius_world / scale

    def forget(self, polydata):
        """Drop cached topology for *polydata* (mesh replaced/removed)."""
        self._topology.pop(id(polydata), None)
        if self._locator_pd is polydata:
            self._locator = None
            self._locator_pd = None
            self._locator_snap = None
