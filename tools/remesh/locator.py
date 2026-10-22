from __future__ import annotations

import numpy as np

from .extract import _bary, _locate, _pt


def binned_locator(Uc, patch, n_patches, fL, fR, fid, seam, tol=1e-9,
                   V=None, F=None, max_dist=None, faces_per_bin=8):
    """Uniform-bin in-patch query, else straight walk across seams."""
    faces_of = [np.flatnonzero(patch == p) for p in range(n_patches)]

    grids = []
    for p in range(n_patches):
        fs = faces_of[p]
        if len(fs) == 0:
            grids.append(None)
            continue
        tri = Uc[fs]
        lo = np.array([tri.real.min(), tri.imag.min()])
        ext = np.array([tri.real.max(), tri.imag.max()]) - lo
        ext = np.maximum(ext, 1e-12)
        nb = max(1, int(round(len(fs) / faces_per_bin)))
        nx = int(max(1, min(round(np.sqrt(nb * ext[0] / max(ext.mean(), 1e-12))),
                            4096)))
        ny = max(1, nb // nx) if nx < nb else 1
        ny = int(min(ny, 4096))
        cell = ext / np.array([nx, ny])
        bl = np.floor((np.stack([tri.real.min(axis=1),
                                 tri.imag.min(axis=1)], axis=1) - lo) / cell)
        bl = np.clip(bl, 0, [nx - 1, ny - 1]).astype(np.int64)
        bh = np.floor((np.stack([tri.real.max(axis=1),
                                 tri.imag.max(axis=1)], axis=1) - lo) / cell)
        bh = np.clip(bh, 0, [nx - 1, ny - 1]).astype(np.int64)
        bins = {}
        for k, f in enumerate(fs):
            for ix in range(bl[k, 0], bh[k, 0] + 1):
                for iy in range(bl[k, 1], bh[k, 1] + 1):
                    bins.setdefault((ix, iy), []).append(int(f))

        def qbin(p, pt, anchor, _p=p, _lo=lo, _cell=cell, _nx=nx, _ny=ny,
                 _bins=bins):
            d = np.floor((np.array([pt.real, pt.imag]) - _lo) / _cell)
            b = _bins.get((int(np.clip(d[0], 0, _nx - 1)),
                           int(np.clip(d[1], 0, _ny - 1))))
            if not b:
                return None
            best = None
            for f in b:
                lam = _bary(Uc[f], pt)
                m = float(lam.min())
                if m < -tol:
                    continue
                lam = np.maximum(lam, 0.0)
                s = lam.sum()
                if s <= 0:
                    continue
                lam = lam / s
                if anchor is not None:
                    dd = float(np.linalg.norm(_pt(lam, V, F, f) - anchor))
                    if max_dist is not None and dd > max_dist:
                        continue
                    sc = (-dd, m)
                else:
                    sc = (0.0, m)
                if best is None or sc > best[0]:
                    best = (sc, f, lam)
            return best

        grids.append(qbin)

    def locate(f0, target, anchor=None):
        r = _locate(Uc, fid, fL, fR, seam, f0, target,
                    anchor=anchor, V=V, F=F, max_dist=max_dist)
        if r is not None:
            return r
        hit = grids[int(patch[f0])](int(patch[f0]), target, anchor) \
            if grids[int(patch[f0])] is not None else None
        if hit is not None:
            _, g, lam = hit
            return g, lam
        return None

    return locate
