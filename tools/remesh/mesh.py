"""Triangle-mesh core: topology, adjacency, normals, areas."""
from __future__ import annotations

import numpy as np


class TriMesh:
    def __init__(self, V: np.ndarray, F: np.ndarray):
        self.V = np.ascontiguousarray(V, dtype=np.float64)
        self.F = np.ascontiguousarray(F, dtype=np.int64)
        if self.V.ndim != 2 or self.V.shape[1] != 3:
            raise ValueError("V must be (n,3)")
        if self.F.ndim != 2 or self.F.shape[1] != 3:
            raise ValueError("F must be (m,3)")
        self._build()

    @classmethod
    def load_obj(cls, path) -> "TriMesh":
        V, F = [], []
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                p = line.split()
                if not p or p[0].startswith("#"):
                    continue
                if p[0] == "v" and len(p) >= 4:
                    V.append([float(p[1]), float(p[2]), float(p[3])])
                elif p[0] == "f" and len(p) >= 4:
                    idx = [int(t.split("/")[0]) - 1 for t in p[1:]]
                    for k in range(1, len(idx) - 1):
                        F.append([idx[0], idx[k], idx[k + 1]])
        return cls(np.array(V, dtype=np.float64), np.array(F, dtype=np.int64))

    def _build(self):
        V, F = self.V, self.F
        n = len(V)

        fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        self.face_areas = 0.5 * np.linalg.norm(fn, axis=1)
        self.face_normals = fn / np.maximum(
            np.linalg.norm(fn, axis=1, keepdims=True), 1e-300)

        self.vertex_normals = np.zeros_like(V)
        for k in range(3):
            np.add.at(self.vertex_normals, F[:, k],
                      self.face_normals * self.face_areas[:, None])
        nrm = np.linalg.norm(self.vertex_normals, axis=1, keepdims=True)
        self.vertex_normals /= np.maximum(nrm, 1e-300)

        self.vertex_areas = np.zeros(n)
        for k in range(3):
            np.add.at(self.vertex_areas, F[:, k], self.face_areas / 3.0)
        self.vertex_areas = np.maximum(self.vertex_areas, 1e-300)

        E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        Fidx = np.tile(np.arange(len(F)), 3)
        lo = np.minimum(E[:, 0], E[:, 1])
        hi = np.maximum(E[:, 0], E[:, 1])
        order = np.lexsort((hi, lo))
        self._edge_lo = lo[order]
        self._edge_hi = hi[order]
        self._edge_face = Fidx[order]
        new_edge = np.empty(len(E), dtype=bool)
        new_edge[0] = True
        new_edge[1:] = (self._edge_lo[1:] != self._edge_lo[:-1]) | \
                       (self._edge_hi[1:] != self._edge_hi[:-1])
        self.edge_ids = np.cumsum(new_edge) - 1
        self.edges = np.stack([self._edge_lo[new_edge],
                               self._edge_hi[new_edge]], axis=1)
        self.edge_face_count = np.bincount(self.edge_ids, minlength=len(self.edges))
        inv = np.empty(len(E), dtype=np.int64)
        inv[order] = np.arange(len(E))
        eid_rec = self.edge_ids[inv]
        self.face_edge_ids = np.stack(
            [eid_rec[:len(F)], eid_rec[len(F):2 * len(F)],
             eid_rec[2 * len(F):]], axis=1)

        vf = np.argsort(F.reshape(-1), kind="stable")
        vs = F.reshape(-1)[vf]
        self._vface_start = np.zeros(n + 1, dtype=np.int64)
        self._vface_start[1:] = np.cumsum(np.bincount(vs, minlength=n))
        self._vface_order = vf // 3
        ve = np.argsort(np.concatenate([self.edges[:, 0], self.edges[:, 1]]),
                        kind="stable")
        vse = np.concatenate([self.edges[:, 0], self.edges[:, 1]])[ve]
        self._vedge_start = np.zeros(n + 1, dtype=np.int64)
        self._vedge_start[1:] = np.cumsum(np.bincount(vse, minlength=n))
        self._vedge_order = ve % len(self.edges)

    def vertex_faces(self, vi) -> np.ndarray:
        s, e = self._vface_start[vi], self._vface_start[vi + 1]
        return self._vface_order[s:e]

    def vertex_edges(self, vi) -> np.ndarray:
        s, e = self._vedge_start[vi], self._vedge_start[vi + 1]
        return self._vedge_order[s:e]

    def edge_faces(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (f_left, f_right) per edge; -1 on boundary."""
        f1 = self._edge_face
        new = np.empty(len(f1), dtype=bool)
        new[0] = True
        new[1:] = (self._edge_lo[1:] != self._edge_lo[:-1]) | \
                  (self._edge_hi[1:] != self._edge_hi[:-1])
        start = np.flatnonzero(new)
        cnt = np.diff(np.append(start, len(f1)))
        f_left = f1[start]
        f_right = np.where(cnt == 2, f1[np.minimum(start + 1, len(f1) - 1)], -1)
        f_right[cnt == 1] = -1
        return f_left, f_right

    @property
    def n_vertices(self):
        return len(self.V)

    @property
    def n_faces(self):
        return len(self.F)

    @property
    def n_edges(self):
        return len(self.edges)
