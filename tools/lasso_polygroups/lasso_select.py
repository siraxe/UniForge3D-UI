"""Screen-space lasso selection for the VTK model viewer.

The viewer feeds this module Qt mouse positions converted to VTK display
coordinates (origin bottom-left).  Three pieces live here:

* LassoOverlay - the near-white rubber-band path drawn while dragging.  It
  is a vtkActor2D whose points are in display coordinates, so it is always
  crisp and always on top of the 3D scene, without touching model materials
  or textures.  For a subtractive lasso the caller passes a red color
  instead.
* combine_selection_cells - unions an additive lasso with the existing face
  selection, or subtracts a red lasso from it.
* faces_in_lasso / build_selection_actors - projection-based face selection.
  Mesh points are projected with the active camera and each face is tested
  against the lasso polygon.  A face is selected when any part of its
  projection is touched by the lasso (a vertex inside, a lasso vertex inside
  the face, or an edge crossing), so large faces that span the lasso without
  having a vertex inside are still caught.

The selection is purely projective, like a 2D lasso: faces on the far side of
the model are selected too.  Their light overlay is hidden by depth testing
until the model is orbited, which is the usual "select through" behaviour.

Only numpy (already a runtime dependency) and VTK are used; no new library is
needed for clipping or projection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import vtk
from vtk.util.numpy_support import (
    numpy_to_vtk,
    numpy_to_vtkIdTypeArray,
    vtk_to_numpy,
)

# Faces get a light neutral tint: it reads as a selection without being
# mistaken for one of the saturated polygroup colours.  The additive lasso
# band is near-white for the same reason; red stays reserved for subtract.
SELECTED_COLOR = (0.88, 0.90, 0.92)
LASSO_COLOR = (0.95, 0.96, 0.98)
LASSO_SUBTRACT_COLOR = (0.96, 0.28, 0.28)   # red = remove from selection

_MAX_TEST_PAIRS = 500_000     # keeps vectorized geometry chunks memory-bounded
_MAX_LASSO_POINTS = 128       # selection polygon detail (drawing stays smooth)
_EPS = 1e-9


@dataclass
class FaceSelection:
    """Face ids selected on one source actor (ids are original cell ids)."""

    actor: vtk.vtkActor
    cell_ids: np.ndarray


def combine_selection_cells(existing, incoming, subtract: bool = False):
    """Union (additive lasso) or difference (red lasso) of cell-id arrays."""
    incoming = np.asarray(incoming, dtype=np.int64)
    if existing is None:
        return (np.zeros(0, dtype=np.int64) if subtract
                else np.unique(incoming))
    existing = np.asarray(existing, dtype=np.int64)
    if subtract:
        return np.setdiff1d(existing, incoming, assume_unique=True)
    return np.union1d(existing, incoming)


class LassoOverlay:
    """A polyline drawn in display coordinates on top of the 3D renderer."""

    def __init__(self, renderer):
        self._renderer = renderer
        self._points = []

        self._polydata = vtk.vtkPolyData()
        self._polydata.SetPoints(vtk.vtkPoints())
        self._polydata.SetLines(vtk.vtkCellArray())

        coordinate = vtk.vtkCoordinate()
        coordinate.SetCoordinateSystemToDisplay()
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(self._polydata)
        mapper.SetTransformCoordinate(coordinate)

        self._actor = vtk.vtkActor2D()
        self._actor.SetMapper(mapper)
        prop = self._actor.GetProperty()
        prop.SetColor(*LASSO_COLOR)
        prop.SetLineWidth(2.0)
        prop.SetOpacity(1.0)
        self._actor.PickableOff()
        self._actor.SetVisibility(False)
        # VTK 9.7 has no vtkRenderer.AddActor2D; AddViewProp handles 2D props.
        renderer.AddViewProp(self._actor)

    @property
    def active(self) -> bool:
        return bool(self._points)

    @property
    def points(self) -> list:
        return list(self._points)

    def begin(self, x: float, y: float, color=None):
        self.set_color(color)
        self._points = [(float(x), float(y))]
        self._rebuild()
        self._actor.SetVisibility(True)

    def set_color(self, color=None):
        """Recolour the live band (near-white = add, red = subtract)."""
        self._actor.GetProperty().SetColor(
            *(LASSO_COLOR if color is None else color))

    def add(self, x: float, y: float) -> bool:
        """Append a point; returns True when the path actually changed."""
        if not self._points:
            return False
        px, py = self._points[-1]
        # 2px spacing keeps the polygon small enough for fast selection and
        # still smooth at normal mouse-drag speeds.
        if (x - px) ** 2 + (y - py) ** 2 < 4.0:
            return False
        self._points.append((float(x), float(y)))
        self._rebuild()
        return True

    def end(self) -> list:
        """Hide the band and return its final points (display coordinates)."""
        points = self.points
        self._reset()
        return points

    def cancel(self):
        self._reset()

    def _reset(self):
        self._points = []
        self._polydata.GetPoints().Reset()
        self._polydata.GetLines().Reset()
        self._polydata.Modified()
        self._actor.SetVisibility(False)

    def _rebuild(self):
        # Close the visual loop while dragging; the selection polygon is
        # implicitly closed by the geometry tests.
        path = self._points + [self._points[0]]
        coordinates = np.asarray(path, dtype=np.float64)
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(
            np.column_stack((coordinates, np.zeros(len(coordinates)))),
            deep=True,
        ))
        ids = np.arange(len(coordinates), dtype=np.int64)
        lines = vtk.vtkCellArray()
        lines.SetData(
            numpy_to_vtkIdTypeArray(np.array([0, len(ids)], dtype=np.int64),
                                    deep=True),
            numpy_to_vtkIdTypeArray(ids, deep=True),
        )
        self._polydata.SetPoints(points)
        self._polydata.SetLines(lines)
        self._polydata.Modified()


def _matrix_as_numpy(matrix) -> np.ndarray:
    return np.array(
        [[matrix.GetElement(row, col) for col in range(4)] for row in range(4)],
        dtype=np.float64,
    )


def _viewport(renderer):
    size = renderer.GetSize()
    if size[0] <= 0 or size[1] <= 0:
        return None, size
    camera = renderer.GetActiveCamera()
    near, far = camera.GetClippingRange()
    if not (np.isfinite(near) and np.isfinite(far)
            and near > 0.0 and far > near):
        near, far = -1.0, 1.0
    matrix = camera.GetCompositeProjectionTransformMatrix(
        renderer.GetTiledAspectRatio(), near, far)
    return _matrix_as_numpy(matrix), size


def _project_actor(actor, view_projection, size):
    """Actor-local points -> display coordinates, plus a w>0 visibility mask."""
    mapper = actor.GetMapper()
    polydata = mapper.GetInput() if mapper is not None else None
    if polydata is None or polydata.GetPoints() is None:
        return None, None
    local = vtk_to_numpy(polydata.GetPoints().GetData())
    if len(local) == 0:
        return None, None

    homogeneous = np.ones((len(local), 4), dtype=np.float64)
    homogeneous[:, :3] = local
    combined = view_projection @ _matrix_as_numpy(actor.GetMatrix())
    clip = homogeneous @ combined.T
    w = clip[:, 3]
    visible = w > _EPS

    screen = np.zeros((len(local), 2), dtype=np.float64)
    if visible.any():
        ndc = clip[visible, :2] / w[visible, None]
        screen[visible] = (ndc + 1.0) * 0.5 * np.asarray(size, dtype=float)
    return screen, visible


def _poly_cells(polydata):
    """(offsets, connectivity) arrays for the polygon cells, or None."""
    polys = polydata.GetPolys()
    if polys is None or polys.GetNumberOfCells() == 0:
        return None
    offsets = vtk_to_numpy(polys.GetOffsetsArray()).astype(np.int64,
                                                           copy=False)
    connectivity = vtk_to_numpy(polys.GetConnectivityArray()).astype(
        np.int64, copy=False)
    if len(offsets) < 2 or len(connectivity) == 0:
        return None
    return offsets, connectivity


def _rdp_simplify(points, epsilon: float = 1.0) -> np.ndarray:
    """Ramer-Douglas-Peucker path simplification (keeps first/last)."""
    count = len(points)
    if count < 3:
        return points
    keep = np.zeros(count, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, count - 1)]
    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        start = points[first]
        segment = points[last] - start
        length_sq = float(segment @ segment)
        middle = points[first + 1:last] - start
        if length_sq <= _EPS:
            distances = np.linalg.norm(middle, axis=1)
        else:
            distances = (np.abs(_cross2(segment[None, :], middle))
                         / np.sqrt(length_sq))
        farthest = int(np.argmax(distances))
        if distances[farthest] > epsilon:
            split = first + 1 + farthest
            keep[split] = True
            stack.append((first, split))
            stack.append((split, last))
    return points[keep]


def _clean_lasso(points):
    """Drop duplicate segments and simplify; return the open polygon."""
    if points is None or len(points) < 3:
        return None
    poly = np.asarray(points, dtype=np.float64)
    keep = [0]
    for i in range(1, len(poly)):
        if np.hypot(*(poly[i] - poly[keep[-1]])) > 1e-6:
            keep.append(i)
    poly = poly[keep]
    if len(poly) >= 2 and np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    if len(poly) < 3:
        return None
    # The visual path keeps every point; the selection polygon only needs
    # sub-pixel accuracy.  Simplifying here keeps hit-testing interactive
    # even for a slowly drawn, several-hundred-point lasso.
    poly = _rdp_simplify(poly)
    if len(poly) > _MAX_LASSO_POINTS:
        indices = np.linspace(0, len(poly) - 1, _MAX_LASSO_POINTS)
        poly = poly[np.unique(np.rint(indices).astype(np.int64))]
    return poly if len(poly) >= 3 else None


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def point_in_polygon(points, polygon) -> np.ndarray:
    """Vectorized even-odd ray casting for N points against one polygon."""
    points = np.asarray(points, dtype=np.float64)
    polygon = np.asarray(polygon, dtype=np.float64)
    inside = np.zeros(len(points), dtype=bool)
    if len(points) == 0 or len(polygon) < 3:
        return inside

    low = polygon.min(axis=0)
    high = polygon.max(axis=0)
    px, py = points[:, 0], points[:, 1]
    candidates = ((px >= low[0]) & (px <= high[0])
                  & (py >= low[1]) & (py <= high[1]))
    indices = np.flatnonzero(candidates)
    if indices.size == 0:
        return inside

    x = px[indices]
    y = py[indices]
    x1 = polygon[:, 0]
    y1 = polygon[:, 1]
    x2 = np.roll(x1, -1)
    y2 = np.roll(y1, -1)
    result = np.zeros(indices.size, dtype=bool)
    for ax, ay, bx, by in zip(x1, y1, x2, y2):
        if ay == by:
            continue
        crosses = (((ay > y) != (by > y))
                   & (x < (bx - ax) * (y - ay) / (by - ay) + ax))
        result ^= crosses
    inside[indices] = result
    return inside


def _lasso_vertex_inside_triangles(tri_screen, poly):
    """(C,) bool: does any lasso vertex fall inside each screen triangle?"""
    count = len(tri_screen)
    result = np.zeros(count, dtype=bool)
    if count == 0 or len(poly) == 0:
        return result

    polygon = poly[None, :, :]
    chunk = max(1, _MAX_TEST_PAIRS // max(1, len(poly)))
    for start in range(0, count, chunk):
        tris = tri_screen[start:start + chunk]
        a = tris[:, 0][:, None, :]
        b = tris[:, 1][:, None, :]
        c = tris[:, 2][:, None, :]
        d1 = _cross2(b - a, polygon - a)
        d2 = _cross2(c - b, polygon - b)
        d3 = _cross2(a - c, polygon - c)
        inside = (((d1 >= 0.0) & (d2 >= 0.0) & (d3 >= 0.0))
                  | ((d1 <= 0.0) & (d2 <= 0.0) & (d3 <= 0.0)))
        result[start:start + chunk] = inside.any(axis=1)
    return result


def _triangle_edges_cross_polygon(tri_screen, poly):
    """(C,) bool: does any triangle edge cross any lasso edge?"""
    count = len(tri_screen)
    result = np.zeros(count, dtype=bool)
    if count == 0 or len(poly) < 2:
        return result

    a = tri_screen[:, 0]
    b = tri_screen[:, 1]
    c = tri_screen[:, 2]
    ax = np.stack((a[:, 0], b[:, 0], c[:, 0]), axis=1)
    ay = np.stack((a[:, 1], b[:, 1], c[:, 1]), axis=1)
    bx = np.stack((b[:, 0], c[:, 0], a[:, 0]), axis=1)
    by = np.stack((b[:, 1], c[:, 1], a[:, 1]), axis=1)
    rx = bx - ax
    ry = by - ay

    p = np.asarray(poly, dtype=np.float64)
    q = np.roll(p, -1, axis=0)
    sx = q[:, 0] - p[:, 0]
    sy = q[:, 1] - p[:, 1]

    chunk = max(1, _MAX_TEST_PAIRS // max(1, 3 * len(p)))
    for start in range(0, count, chunk):
        x0 = ax[start:start + chunk][:, :, None]
        y0 = ay[start:start + chunk][:, :, None]
        rxs = rx[start:start + chunk][:, :, None]
        rys = ry[start:start + chunk][:, :, None]
        sxs = sx[None, None, :]
        sys_ = sy[None, None, :]

        den = rxs * sys_ - rys * sxs
        qx = p[None, None, :, 0] - x0
        qy = p[None, None, :, 1] - y0
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (qx * sys_ - qy * sxs) / den
            u = (qx * rys - qy * rxs) / den
        hit = ((np.abs(den) > _EPS)
               & (t >= -_EPS) & (t <= 1.0 + _EPS)
               & (u >= -_EPS) & (u <= 1.0 + _EPS))
        result[start:start + chunk] = hit.any(axis=(1, 2))
    return result


def _triangles_touch_polygon(tri_screen, poly):
    """(C,) bool: triangle / lasso polygon intersection tests.

    The caller normally already handled vertices inside the lasso, but the
    checks are repeated so this helper is correct on its own.
    """
    count = len(tri_screen)
    result = np.zeros(count, dtype=bool)
    if count == 0 or len(poly) < 3:
        return result

    a = tri_screen[:, 0]
    b = tri_screen[:, 1]
    c = tri_screen[:, 2]
    area = _cross2(b - a, c - a)
    usable = np.abs(area) > 1e-12
    if not usable.any():
        return result

    ids = np.flatnonzero(usable)
    tris = tri_screen[ids]
    hit = _lasso_vertex_inside_triangles(tris, poly)
    hit |= _triangle_edges_cross_polygon(tris, poly)
    result[ids] = hit
    return result


def _segments_intersect_any(first, second) -> bool:
    """True when any edge of two simple 2D polygons crosses."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if len(first) < 2 or len(second) < 2:
        return False
    first_next = np.roll(first, -1, axis=0)
    second_next = np.roll(second, -1, axis=0)
    rx = first_next[:, 0] - first[:, 0]
    ry = first_next[:, 1] - first[:, 1]
    sx = second_next[:, 0] - second[:, 0]
    sy = second_next[:, 1] - second[:, 1]

    den = rx[:, None] * sy[None, :] - ry[:, None] * sx[None, :]
    qx = second[None, :, 0] - first[:, None, 0]
    qy = second[None, :, 1] - first[:, None, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qx * sy[None, :] - qy * sx[None, :]) / den
        u = (qx * ry[:, None] - qy * rx[:, None]) / den
    hit = ((np.abs(den) > _EPS)
           & (t >= -_EPS) & (t <= 1.0 + _EPS)
           & (u >= -_EPS) & (u <= 1.0 + _EPS))
    return bool(hit.any())


def _polygon_touches_polygon(cell_screen, poly) -> bool:
    """Exact-enough intersection of one mesh face with the lasso polygon."""
    if len(cell_screen) < 3:
        return False
    if point_in_polygon(cell_screen, poly).any():
        return True
    if point_in_polygon(poly, cell_screen).any():
        return True
    return _segments_intersect_any(cell_screen, poly)


def _select_cells(screen, visible, offsets, connectivity, poly,
                  poly_low, poly_high):
    """Boolean mask of polygon cells touched by the lasso."""
    cell_count = len(offsets) - 1
    selected = np.zeros(cell_count, dtype=bool)
    if cell_count <= 0 or len(connectivity) == 0:
        return selected

    sizes = np.diff(offsets)
    starts = offsets[:-1]

    # Fast, vectorized pass: any projected vertex inside.
    inside_points = point_in_polygon(screen, poly) & visible
    vertex_hit = np.maximum.reduceat(
        inside_points[connectivity].astype(np.uint8), starts) > 0
    selected |= vertex_hit

    visible_flat = visible[connectivity]
    counts = np.add.reduceat(visible_flat.astype(np.int64), starts)
    any_visible = counts > 0
    if not any_visible.any():
        return selected

    x = np.where(visible_flat, screen[connectivity, 0], np.inf)
    y = np.where(visible_flat, screen[connectivity, 1], np.inf)
    min_x = np.minimum.reduceat(x, starts)
    min_y = np.minimum.reduceat(y, starts)
    x = np.where(visible_flat, screen[connectivity, 0], -np.inf)
    y = np.where(visible_flat, screen[connectivity, 1], -np.inf)
    max_x = np.maximum.reduceat(x, starts)
    max_y = np.maximum.reduceat(y, starts)
    bbox_hit = (any_visible
                & (max_x >= poly_low[0]) & (min_x <= poly_high[0])
                & (max_y >= poly_low[1]) & (min_y <= poly_high[1]))

    # Exact tests only for boundary cells that survived the cheap pass.
    candidates = np.flatnonzero(bbox_hit & ~selected)
    if candidates.size == 0:
        return selected
    # A face crossing the near plane has invalid vertex projections; its
    # visible-vertex check above still applies, but skip edge tests.
    candidates = candidates[counts[candidates] == sizes[candidates]]
    if candidates.size == 0:
        return selected

    triangles = candidates[sizes[candidates] == 3]
    if triangles.size:
        gather = starts[triangles][:, None] + np.arange(3, dtype=np.int64)
        hit = _triangles_touch_polygon(screen[connectivity[gather]], poly)
        selected[triangles[hit]] = True

    general = candidates[sizes[candidates] != 3]
    for cell_id in general:
        ids = connectivity[offsets[cell_id]:offsets[cell_id + 1]]
        if _polygon_touches_polygon(screen[ids], poly):
            selected[cell_id] = True
    return selected


def faces_in_lasso(renderer, actors, lasso_points) -> list:
    """Return a FaceSelection for every actor with selected faces."""
    poly = _clean_lasso(lasso_points)
    if poly is None or not actors:
        return []
    view_projection, size = _viewport(renderer)
    if view_projection is None:
        return []

    poly_low = poly.min(axis=0)
    poly_high = poly.max(axis=0)
    selections = []
    for actor in actors:
        mapper = actor.GetMapper()
        polydata = mapper.GetInput() if mapper is not None else None
        if polydata is None:
            continue
        cells = _poly_cells(polydata)
        if cells is None:
            continue
        screen, visible = _project_actor(actor, view_projection, size)
        if screen is None or not visible.any():
            continue
        selected = _select_cells(screen, visible, cells[0], cells[1], poly,
                                 poly_low, poly_high)
        cell_ids = np.flatnonzero(selected)
        if cell_ids.size:
            selections.append(FaceSelection(actor, cell_ids))
    return selections


def build_cell_array(polydata, cell_ids):
    """A vtkCellArray holding only *cell_ids* of *polydata*.

    Connectivity still refers to the original point ids, so the result can be
    attached to a vtkPolyData that shares the source points.  Returns None
    for empty input or geometry that has no polygons.
    """
    cells = _poly_cells(polydata)
    if cells is None:
        return None
    cell_ids = np.asarray(cell_ids, dtype=np.int64)
    if cell_ids.size == 0:
        return None
    offsets, connectivity = cells
    cell_count = len(offsets) - 1
    cell_ids = cell_ids[(cell_ids >= 0) & (cell_ids < cell_count)]
    if cell_ids.size == 0:
        return None
    sizes = np.diff(offsets)[cell_ids]
    starts = offsets[cell_ids]

    new_offsets = np.empty(len(cell_ids) + 1, dtype=np.int64)
    new_offsets[0] = 0
    np.cumsum(sizes, out=new_offsets[1:])
    if np.all(sizes == sizes[0]):
        gather = starts[:, None] + np.arange(int(sizes[0]), dtype=np.int64)
        new_connectivity = connectivity[gather].ravel()
    else:
        parts = [connectivity[starts[i]:starts[i] + sizes[i]]
                 for i in range(len(cell_ids))]
        new_connectivity = (np.concatenate(parts) if parts
                            else np.zeros(0, dtype=np.int64))

    cell_array = vtk.vtkCellArray()
    cell_array.SetData(
        numpy_to_vtkIdTypeArray(new_offsets, deep=True),
        numpy_to_vtkIdTypeArray(np.ascontiguousarray(new_connectivity),
                                deep=True),
    )
    return cell_array


def build_selection_actors(selections, backface_cull: bool = True,
                           color=SELECTED_COLOR,
                           polygon_offset=(-1.0, -1.0)) -> list:
    """Build coloured overlay actors for the selected faces.

    Each overlay shares the source actor's points (including normals) and
    transform, but owns a cell array containing only the selected faces.  A
    polygon offset lifts it off the source surface to avoid z-fighting while
    keeping normal depth occlusion from the rest of the scene.  Polygroup
    overlays pass their own *color* and a smaller offset so the selection
    highlight still wins when both cover the same face.
    """
    overlays = []
    for selection in selections:
        source = selection.actor
        cell_ids = np.asarray(selection.cell_ids, dtype=np.int64)
        mapper = source.GetMapper()
        polydata = mapper.GetInput() if mapper is not None else None
        if polydata is None or cell_ids.size == 0:
            continue
        cell_array = build_cell_array(polydata, cell_ids)
        if cell_array is None:
            continue

        highlight = vtk.vtkPolyData()
        highlight.SetPoints(polydata.GetPoints())
        highlight.GetPointData().ShallowCopy(polydata.GetPointData())
        highlight.SetPolys(cell_array)

        highlight_mapper = vtk.vtkPolyDataMapper()
        highlight_mapper.SetInputData(highlight)
        highlight_mapper.SetScalarVisibility(False)
        highlight_mapper.SetResolveCoincidentTopologyToPolygonOffset()
        highlight_mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(
            *polygon_offset)

        actor = vtk.vtkActor()
        actor.SetMapper(highlight_mapper)
        matrix = vtk.vtkMatrix4x4()
        matrix.DeepCopy(source.GetMatrix())
        actor.SetUserMatrix(matrix)
        prop = actor.GetProperty()
        prop.SetInterpolationToPBR()
        prop.SetColor(*color)
        prop.SetMetallic(0.0)
        prop.SetRoughness(0.55)
        prop.SetAmbient(0.30)
        prop.SetDiffuse(0.90)
        if backface_cull:
            prop.BackfaceCullingOn()
        else:
            prop.BackfaceCullingOff()
        actor.PickableOff()
        actor.SetVisibility(source.GetVisibility())
        actor._selection_source = source
        actor._selection_count = len(cell_ids)
        overlays.append(actor)
    return overlays
