"""Batched GPU fold-removal (untangle) for the integer UV layout trials."""
from __future__ import annotations

import os

import numpy as np

_TORCH = None
_HAS_CUDA = False
_CHECKED = False


def _init_torch():
    global _TORCH, _HAS_CUDA, _CHECKED
    if _CHECKED:
        return
    _CHECKED = True
    if os.environ.get("REMESH_NO_GPU", "").strip().lower() in \
            ("1", "true", "yes", "on"):
        return
    try:
        import torch
        _TORCH = torch
        _HAS_CUDA = torch.cuda.is_available()
    except Exception:
        _TORCH = None
        _HAS_CUDA = False


def gpu_available() -> bool:
    """True when CUDA torch is importable and a device is present."""
    _init_torch()
    return _HAS_CUDA


def untangle_batch(uvres, jobs, max_restarts=4, maxiter=1500,
                   log=None, on_iter=None, cancel_check=None):
    """Fold-remove several trial layouts at once on the GPU.

    ``jobs`` is a list of ``{"label", "U0" (complex n_uv), "eps"}``
    dicts - one per lattice trial, all sharing the same cut layout
    (``uvres``).  Returns one result dict per job mirroring
    :func:`remesh.integer.untangle_uv`'s output.  ``on_iter(label, nit,
    energy, restart)`` is called during optimization; ``cancel_check``
    is polled every optimizer iteration and raises
    :class:`remesh.integer.RemeshCancelled`.
    """
    _init_torch()
    if not _HAS_CUDA:
        raise RuntimeError("CUDA torch unavailable")
    torch = _TORCH
    from .integer import RemeshCancelled, seam_nullspace

    def say(message, level="info"):
        if log is not None:
            log(message, level)

    jobs = list(jobs)
    if not jobs:
        return []

    dev = torch.device("cuda")
    fp32 = os.environ.get("REMESH_GPU_FP32", "").strip().lower() in \
        ("1", "true", "yes", "on")
    rdtype = torch.float32 if fp32 else torch.float64
    cdtype = torch.complex64 if fp32 else torch.complex128
    csize = torch.empty((), dtype=cdtype).element_size()

    corner = np.ascontiguousarray(uvres["corner_uv"], dtype=np.int64)
    roots, rots, pin = seam_nullspace(uvres)
    n_uv = len(roots)
    free = np.unique(roots[~pin])
    nv = len(free)
    col = np.full(n_uv, -1, dtype=np.int64)
    col[~pin] = np.searchsorted(free, roots[~pin])
    valid = col >= 0
    vidx = np.flatnonzero(valid).astype(np.int64)
    vcol = np.ascontiguousarray(col[vidx], dtype=np.int64)

    rots_t = torch.as_tensor(rots).to(device=dev, dtype=cdtype)
    rot_re = rots_t.real.contiguous()
    rot_im = rots_t.imag.contiguous()
    ci0 = torch.as_tensor(corner[:, 0], device=dev)
    ci1 = torch.as_tensor(corner[:, 1], device=dev)
    ci2 = torch.as_tensor(corner[:, 2], device=dev)
    vidx_t = torch.as_tensor(vidx, device=dev)
    vcol_t = torch.as_tensor(vcol, device=dev)

    nF = corner.shape[0]
    T_all = len(jobs)

    ub_all = torch.stack([
        torch.as_tensor(np.asarray(j["U0"], dtype=np.complex128))
        for j in jobs]).to(device=dev, dtype=cdtype)

    eps_np = np.empty(T_all, dtype=np.float64)
    for i, j in enumerate(jobs):
        if j.get("eps") is not None:
            eps_np[i] = float(j["eps"])
        else:
            U0 = np.asarray(j["U0"])
            Uc0 = U0[corner]
            a0 = np.imag(np.conj(Uc0[:, 1] - Uc0[:, 0]) *
                         (Uc0[:, 2] - Uc0[:, 0]))
            eps_np[i] = max(0.01 * float(np.median(np.maximum(a0, 0.0))),
                            1e-9)

    budget = int(os.environ.get("REMESH_GPU_CHUNK_FACES", "12000000"))
    chunk = max(1, min(T_all, budget // max(1, nF)))
    say(f"integer: GPU untangle on {torch.cuda.get_device_name(0)}: "
        f"{T_all} trials x {nF:,} faces, batch {chunk} "
        f"({'fp32' if fp32 else 'fp64'}) ...")

    # L-BFGS history memory guard: 2 buffers (s, y) per pair.
    max_m = 20
    if nv > 0:
        pair_bytes = 2 * chunk * nv * csize
        max_m = int(max(4, min(20, 4e9 // max(1, pair_bytes))))

    results = [None] * T_all
    import time as _time
    t_start = _time.perf_counter()
    tot_iters = [0]
    tot_evals = [0]

    for lo in range(0, T_all, chunk):
        hi = min(lo + chunk, T_all)
        T = hi - lo
        labels = [jobs[i]["label"] for i in range(lo, hi)]
        ubc = ub_all[lo:hi]
        ub_re = ubc.real.contiguous()
        ub_im = ubc.imag.contiguous()
        eps = torch.as_tensor(eps_np[lo:hi], device=dev,
                              dtype=rdtype).view(T, 1)
        eps1 = eps.view(-1)

        # precomputed chain-rotation constants at the free corner slots
        vr = rot_re.index_select(0, vidx_t).contiguous()
        vi = rot_im.index_select(0, vidx_t).contiguous()

        def evaluate(x):
            """Energy/gradient for all trials; mirrors the CPU objective
            in remesh.integer.untangle_uv exactly.  ``x`` is the real
            (T, 2*nv) free-variable vector; returns Uc as real pair."""
            zr = x[:, 0::2]
            zi = x[:, 1::2]
            zcr = torch.zeros((T, n_uv), dtype=rdtype, device=dev)
            zci = torch.zeros((T, n_uv), dtype=rdtype, device=dev)
            zr_v = zr.index_select(1, vcol_t)
            zi_v = zi.index_select(1, vcol_t)
            zcr.index_copy_(1, vidx_t, zr_v)
            zci.index_copy_(1, vidx_t, zi_v)
            ucr = ub_re + rot_re * zcr - rot_im * zci
            uci = ub_im + rot_re * zci + rot_im * zcr
            p0r = ucr.index_select(1, ci0)
            p0i = uci.index_select(1, ci0)
            p1r = ucr.index_select(1, ci1)
            p1i = uci.index_select(1, ci1)
            p2r = ucr.index_select(1, ci2)
            p2i = uci.index_select(1, ci2)
            e1xr = p1r - p0r
            e1xi = p1i - p0i
            e2xr = p2r - p0r
            e2xi = p2i - p0i
            area = e1xr * e2xi - e1xi * e2xr
            d = torch.clamp(eps - area, min=0.0)
            loss = (d * d).sum(dim=1)
            gA = torch.where(d > 0, -2.0 * d, torch.zeros_like(d))
            ge1r = gA * e2xi
            ge1i = -gA * e2xr
            ge2r = -gA * e1xi
            ge2i = gA * e1xr
            gcr = torch.zeros((T, n_uv), dtype=rdtype, device=dev)
            gci = torch.zeros((T, n_uv), dtype=rdtype, device=dev)
            gcr.index_add_(1, ci0, -(ge1r + ge2r))
            gci.index_add_(1, ci0, -(ge1i + ge2i))
            gcr.index_add_(1, ci1, ge1r)
            gci.index_add_(1, ci1, ge1i)
            gcr.index_add_(1, ci2, ge2r)
            gci.index_add_(1, ci2, ge2i)
            # chain back through the corner rotations (gc * conj(rot))
            gvr = gcr.index_select(1, vidx_t)
            gvi = gci.index_select(1, vidx_t)
            gzr = gvr * vr + gvi * vi
            gzi = gvi * vr - gvr * vi
            gzr_all = torch.zeros((T, nv), dtype=rdtype, device=dev)
            gzi_all = torch.zeros((T, nv), dtype=rdtype, device=dev)
            gzr_all.index_add_(1, vcol_t, gzr)
            gzi_all.index_add_(1, vcol_t, gzi)
            g = torch.empty((T, 2 * nv), dtype=rdtype, device=dev)
            g[:, 0::2] = gzr_all
            g[:, 1::2] = gzi_all
            return loss, g, ucr, uci, area

        def rdot(a, b):
            """Real Euclidean dot per trial: (T, 1)."""
            return (a * b).sum(dim=1, keepdim=True)

        # stacked L-BFGS history: slot 0 = oldest pair
        S_buf = torch.zeros((max_m, T, 2 * nv), dtype=rdtype, device=dev)
        Y_buf = torch.zeros((max_m, T, 2 * nv), dtype=rdtype, device=dev)
        RHO_buf = torch.zeros((max_m, T), dtype=rdtype, device=dev)
        MSK_buf = torch.zeros((max_m, T), dtype=torch.bool, device=dev)

        def direction(n_pairs, g, halt):
            """Exact L-BFGS two-loop recursion, batched over trials.

            The recursion is sequential in the pair index (each alpha
            depends on the partially-updated q), so slots iterate in a
            short Python loop while every step itself is one batched
            kernel over all trials.
            """
            q = g
            zero = torch.zeros((T, 1), dtype=rdtype, device=dev)
            alphas = []
            for i in range(n_pairs - 1, -1, -1):
                s_i, y_i = S_buf[i], Y_buf[i]
                alpha = torch.where(MSK_buf[i].view(T, 1),
                                    RHO_buf[i].view(T, 1) * rdot(s_i, q),
                                    zero)
                q = q - alpha * s_i
                alphas.append(alpha)
            if n_pairs:
                s_k, y_k = S_buf[n_pairs - 1], Y_buf[n_pairs - 1]
                gam = torch.where(
                    MSK_buf[n_pairs - 1].view(T, 1),
                    rdot(s_k, y_k) /
                    torch.clamp(rdot(y_k, y_k), min=1e-300),
                    torch.ones((T, 1), dtype=rdtype, device=dev))
                q = gam * q
            for i in range(n_pairs):
                s_i, y_i = S_buf[i], Y_buf[i]
                beta = torch.where(MSK_buf[i].view(T, 1),
                                   RHO_buf[i].view(T, 1) * rdot(y_i, q),
                                   zero)
                q = q + (alphas[n_pairs - 1 - i] - beta) * s_i
            dr = -q
            dr[halt] = 0
            return dr

        x = torch.zeros((T, 2 * nv), dtype=rdtype, device=dev)
        done_t = torch.zeros(T, dtype=torch.bool, device=dev)
        done = np.zeros(T, dtype=bool)
        nit = np.zeros(T, dtype=np.int64)
        nfev = np.zeros(T, dtype=np.int64)
        restarts = np.zeros(T, dtype=np.int64)
        c1 = 1e-4
        max_ls = 12
        stall_limit = 2
        n_pairs = 0
        t_warm = torch.ones((T, 1), dtype=rdtype, device=dev)

        loss, g, ucr, uci, area = evaluate(x)
        nfev += 1

        for restart in range(max_restarts):
            if cancel_check is not None and cancel_check():
                raise RemeshCancelled("gpu untangle cancelled")
            restarts[~done] = restart + 1
            n_pairs = 0
            stalled = torch.zeros(T, dtype=rdtype, device=dev)
            for it in range(maxiter):
                minarea = area.min(dim=1).values
                flips = (area <= 0).sum(dim=1)
                perfect = flips == 0
                if nv > 0:
                    perfect = perfect & (minarea >= 0.5 * eps1)
                newly = (~done_t) & perfect
                if bool(newly.any()):
                    done_t |= newly
                    done = done_t.cpu().numpy()
                halt = done_t | (stalled >= stall_limit)
                if bool(halt.all()):
                    break
                dr = direction(n_pairs, g, halt)
                if not bool((dr != 0).any()):
                    break
                gtd = rdot(g, dr).view(-1)
                prev_loss = loss
                t = torch.clamp(2.0 * t_warm, max=1.0)
                accepted = halt.clone()
                # last state that satisfied Armijo per trial; a trial that
                # never satisfies it keeps its previous state (no forced
                # garbage steps).
                x_ok, l_ok, g_ok = x, loss, g
                ur_ok, ui_ok, a_ok = ucr, uci, area
                for ls in range(max_ls + 1):
                    x_try = torch.where(halt.view(T, 1), x, x + t * dr)
                    l_t, g_t, ur_t, ui_t, a_t = evaluate(x_try)
                    nfev[~done] += 1
                    nfev_now = int((~done).sum())
                    # strong Wolfe: Armijo decrease + curvature condition;
                    # like L-BFGS-B, failing to satisfy both ends the trial
                    gtd_t = rdot(g_t, dr).view(-1)
                    ok = halt | (torch.isfinite(l_t) &
                                 (l_t <= loss + c1 * t.view(-1) * gtd) &
                                 (gtd_t.abs() <= 0.9 * gtd.abs()))
                    fresh = ok & ~accepted
                    if bool(fresh.any()):
                        sel = fresh.view(T, 1)
                        x_ok = torch.where(sel, x_try, x_ok)
                        ur_ok = torch.where(sel, ur_t, ur_ok)
                        ui_ok = torch.where(sel, ui_t, ui_ok)
                        a_ok = torch.where(sel, a_t, a_ok)
                        l_ok = torch.where(fresh, l_t, l_ok)
                        g_ok = torch.where(sel, g_t, g_ok)
                    accepted |= ok
                    if bool(accepted.all()) or ls == max_ls:
                        break
                    t = torch.where(ok.view(T, 1), t, t * 0.5)
                s = x_ok - x
                y = g_ok - g
                x, loss, g = x_ok, l_ok, g_ok
                ucr, uci, area = ur_ok, ui_ok, a_ok
                t_warm = t
                # per-trial stall tracking: like scipy's L-BFGS-B, a trial
                # whose line search keeps failing - or that has converged
                # to numerical noise - is left alone.
                converged = (loss >= prev_loss) | \
                    ((prev_loss - loss) <= 1e-13 *
                     torch.clamp(prev_loss, min=1e-300))
                progressed = ok & ~halt & ~converged
                stalled = torch.where(progressed, torch.zeros_like(stalled),
                                      torch.where(~halt, stalled + 1.0,
                                                  stalled))
                sy = rdot(s, y).view(-1)
                ss = rdot(s, s).view(-1)
                yy = rdot(y, y).view(-1)
                vmask = sy > 1e-10 * torch.sqrt(torch.clamp(ss * yy,
                                                            min=0.0))
                if bool(vmask.any()):
                    sy_safe = torch.where(vmask, sy, torch.ones_like(sy))
                    slot = n_pairs % max_m
                    S_buf[slot], Y_buf[slot] = s, y
                    RHO_buf[slot] = 1.0 / sy_safe
                    MSK_buf[slot] = vmask
                    n_pairs = min(n_pairs + 1, max_m)
                nit[~done] += 1
                tot_iters[0] += int((~done).sum())
                tot_evals[0] += int(nfev_now)
                if on_iter is not None and it % 25 == 0:
                    l_cpu = loss.cpu().numpy()
                    halt_np = done | (stalled.cpu().numpy() >= stall_limit)
                    for i in range(T):
                        if not halt_np[i]:
                            on_iter(labels[i], int(nit[i]),
                                    float(l_cpu[i]), int(restarts[i]))

            minarea = area.min(dim=1).values
            flips = (area <= 0).sum(dim=1)
            perfect = (flips == 0) & (minarea >= 0.5 * eps1)
            newly = (~done_t) & perfect
            if bool(newly.any()):
                done_t |= newly
                done = done_t.cpu().numpy()
            if done.all() or restart == max_restarts - 1:
                break
            # Re-anchor active trials at their current layout (the CPU
            # path restarts L-BFGS from U with fresh memory), keep the
            # finished ones frozen.
            with torch.no_grad():
                act = torch.nonzero(~done_t, as_tuple=False).view(-1)
                if len(act) == T:
                    ub_re, ub_im = ucr, uci
                else:
                    ub_re = ub_re.clone()
                    ub_im = ub_im.clone()
                    ub_re[act] = ucr[act]
                    ub_im[act] = uci[act]
                x = torch.zeros((T, 2 * nv), dtype=rdtype, device=dev)
                t_warm = torch.ones((T, 1), dtype=rdtype, device=dev)
            loss, g, ucr, uci, area = evaluate(x)
            nfev[~done] += 1

        Uc_np = (ucr + 1j * uci).cpu().numpy()
        area_np = area.cpu().numpy()
        for i in range(T):
            a_i = area_np[i]
            results[lo + i] = {
                "U": Uc_np[i], "flips": int((a_i <= 0).sum()),
                "min_area": float(a_i.min()), "eps": float(eps_np[lo + i]),
                "nit": int(nit[i]), "nfev": int(nfev[i]),
                "restarts": int(restarts[i]),
                "roots": roots, "rots": rots, "pinned": pin}
        if any(r["flips"] == 0 and r["min_area"] >= 0.5 * r["eps"]
               for r in results[lo:hi]):
            say("integer: GPU untangle found a fold-free trial; "
                "skipping remaining batches")
            break

    say(f"integer: GPU untangle done in {_time.perf_counter() - t_start:.1f}s "
        f"({tot_iters[0]:,} batch-iters, {tot_evals[0]:,} batch-evals)")
    return results


__all__ = ["gpu_available", "untangle_batch"]
