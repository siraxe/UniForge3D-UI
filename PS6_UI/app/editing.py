"""In-memory edit stack: pending geometry, Save/Bake, Split-and-save."""

from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

import viewer as viewer_mod
from dialogs import SplitNameDialog
from paths import _edit_output_path, _paths_equal
from utils.mesh import as_triangles

class EditSession(QObject):
    """Owns the unsaved edit stack and writes it when the user saves."""

    log = Signal(str, str)
    failed = Signal(str, str)
    changed = Signal()
    saved = Signal(object, bool)        # output path, viewer loaded it
    folder_changed = Signal(object)

    def __init__(self, viewer, jobs, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.jobs = jobs
        self.pending: dict | None = None
        self._smooth_bake_logged = False
        self._smooth_dirty = False   # pending exists but has no arrays yet

    @property
    def has_pending(self) -> bool:
        return self.pending is not None

    # -- pending stack -------------------------------------------------------
    def arrays_for(self, target: Path):
        """The current unsaved arrays when they belong to *target*."""
        self._materialize()
        edit = self.pending
        if edit is None:
            return None, None
        try:
            edit_target = Path(edit["target_path"])
        except (KeyError, TypeError):
            return None, None
        try:
            same = edit_target.resolve() == Path(target).resolve()
        except OSError:
            same = edit_target == Path(target)
        if not same:
            return None, None
        return edit.get("vertices"), edit.get("faces")

    def _materialize(self):
        """Snapshot the displayed mesh into the pending edit (lazy capture).

        A stroke end only marks the edit dirty: extracting the whole
        displayed mesh (vertices + fan-triangulated faces) costs hundreds
        of milliseconds on dense models, and nobody needs the arrays at
        stroke time - the displayed geometry *is* the truth until then.
        This fills the pending edit in the instant the arrays are actually
        requested (Save/Bake, or another operation that chains onto the
        unsaved edit).
        """
        if not self._smooth_dirty or self.pending is None:
            return
        geometry = self.viewer.edited_geometry()
        if geometry is None:
            return             # save() reports the missing geometry later
        self.pending["vertices"], self.pending["faces"] = geometry
        self._smooth_dirty = False

    def capture_smooth_brush(self):
        """Queue the displayed (smooth-brush edited) mesh for Bake.

        The viewer emits smoothEdited after every stroke and every undo.
        Deliberately cheap: it only marks the pending edit dirty (see
        _materialize) and keeps the history labels - the full-mesh
        extraction no longer runs on the release of every stroke, which
        is what used to hitch the UI on dense models.
        """
        if self.jobs.busy:
            return
        target = self.viewer.current_path
        if target is None:
            return
        previous = self.pending
        try:
            same = (previous is not None
                    and Path(previous["target_path"]).resolve()
                    == target.resolve())
        except (KeyError, TypeError, OSError):
            same = False
        history = (list(previous.get("history")
                        or [previous.get("mode_label", "Edit")])
                   if same else [])
        if not history or history[-1] != "Smooth brush":
            history.append("Smooth brush")
        self.pending = {
            "target_path": str(target),
            "mode_label": "Smooth brush",
            "history": history,
        }
        # a previous materialized snapshot is now stale
        if previous is not None:
            previous.pop("vertices", None)
            previous.pop("faces", None)
        self._smooth_dirty = True
        self.changed.emit()
        if not same or not self._smooth_bake_logged:
            self.log.emit("Smooth brush edits are pending - press Bake "
                          "changes to write them into the model file.",
                          "info")
        self._smooth_bake_logged = True

    def discard(self, reason: str = ""):
        """Drop the unsaved edit (e.g. another model loads)."""
        if self.pending is None:
            return
        self.pending = None
        self._smooth_bake_logged = False
        self._smooth_dirty = False
        self.changed.emit()
        if reason:
            self.log.emit(f"Edit: unsaved edit discarded ({reason}).", "warn")

    def retarget(self, old: Path, new: Path):
        """Retarget the pending edit after its model file was renamed."""
        if self.pending is None:
            return
        try:
            target = Path(self.pending["target_path"])
        except (KeyError, TypeError):
            return
        if _paths_equal(target, old):
            self.pending["target_path"] = new
            self.changed.emit()

    # -- running operations --------------------------------------------------
    def _target(self, label: str):
        if self.jobs.busy:
            return None
        loaded = self.viewer.current_path
        if loaded is None or not loaded.is_file():
            self.log.emit(f"{label}: no model is loaded.", "warn")
            return None
        return loaded

    def run_reproject(self, *, mode, mode_label, dist, strength, blur,
                      shell, detail, subdiv):
        """Copy the ghosted donor mesh's detail onto the displayed mesh."""
        loaded = self._target("Reproject")
        if loaded is None:
            return
        source = self.viewer.ghost_path
        if source is None:
            self.log.emit("Reproject: no donor mesh picked. Middle-click an "
                          "entry in the Outliner column to ghost it; the "
                          "ghosted mesh provides the detail to reproject.",
                          "warn")
            return
        if not source.is_file():
            self.log.emit(f"Reproject: the ghosted donor {source.name} no "
                          "longer exists on disk.", "warn")
            return
        if _paths_equal(source, loaded):
            self.log.emit("Reproject: the displayed mesh itself is the ghost; "
                          "ghost a different mesh as the donor.", "warn")
            return

        target_vertices, target_faces = as_triangles(
            *self.arrays_for(loaded))
        if target_vertices is not None:
            self.log.emit("Reproject: using the current unsaved in-memory "
                          "edit as the target, so earlier changes are kept.",
                          "info")
        if mode == "higher_subdiv" and subdiv >= 3:
            self.log.emit("Reproject: higher subdiv x3 multiplies the face "
                          "count by 64; this can be very memory-heavy.", "warn")
        detail_text = (f", subdiv x{subdiv}" if mode == "higher_subdiv"
                       else f", dist {dist:g}")
        self.log.emit(f"Reproject ({mode_label}) from {source.name} onto "
                      f"{loaded.name} (strength {strength * 100:.0f}%, "
                      f"blur {blur}, shell {shell:g}{detail_text}); the result "
                      "stays in memory until Save.", "info")
        self.jobs.reproject.start(
            loaded, source, mode=mode, dist=dist, strength=strength,
            pa_blur=blur, projection_shell=shell,
            detail_iterations=detail, subdiv_levels=subdiv,
            target_vertices=target_vertices, target_faces=target_faces)
        self.changed.emit()

    def run_polish(self, *, amount, preserve_peaks):
        loaded = self._target("Polish")
        if loaded is None:
            return
        if amount <= 0:
            self.log.emit("Polish: amount is 0%; nothing to do.", "warn")
            return
        vertices, faces = as_triangles(*self.arrays_for(loaded))
        if vertices is not None:
            self.log.emit("Polish: using the current unsaved in-memory edit "
                          "as input; earlier changes are kept.", "info")
        self.log.emit(f"Polish on {loaded.name}: amount {amount}%"
                      + (", experimental preserve-peaks path"
                         if preserve_peaks else "")
                      + "; the result stays in memory until Save.", "info")
        self.jobs.polish.start(loaded, amount=amount,
                               preserve_peaks=preserve_peaks,
                               vertices=vertices, faces=faces)
        self.changed.emit()

    def run_remesh(self, *, target_quads, symmetry, freeze_borders,
                   var_density_ratio):
        loaded = self._target("Remesh")
        if loaded is None:
            return
        vertices, faces = as_triangles(*self.arrays_for(loaded))
        if vertices is not None:
            self.log.emit("Remesh: using the current unsaved in-memory edit "
                          "as input; earlier changes are kept.", "info")
        target_text = (f"{target_quads:,} quads" if target_quads > 0
                       else "auto quads")
        self.log.emit(f"Remesh on {loaded.name}: {target_text}, density "
                      f"{var_density_ratio:g}; the output topology changes and "
                      "the result stays in memory until Save.", "warn")
        self.jobs.remesh.start(
            loaded, target_quads=target_quads, symmetry=symmetry,
            freeze_borders=freeze_borders,
            var_density_ratio=var_density_ratio,
            vertices=vertices, faces=faces)
        self.changed.emit()

    def run_decimate(self, *, target_tris):
        loaded = self._target("Decimate")
        if loaded is None:
            return
        vertices, faces = as_triangles(*self.arrays_for(loaded))
        if vertices is not None:
            self.log.emit("Decimate: using the current unsaved in-memory edit "
                          "as input; earlier changes are kept.", "info")
        self.log.emit(f"Decimate on {loaded.name}: target {target_tris:,} "
                      "triangles; the output topology changes and the result "
                      "stays in memory until Save.", "info")
        self.jobs.decimate.start(loaded, target_tris=target_tris,
                                 vertices=vertices, faces=faces)
        self.changed.emit()

    def run_subdivide(self):
        loaded = self._target("Subdivide")
        if loaded is None:
            return
        vertices, faces = self.arrays_for(loaded)
        if vertices is not None:
            self.log.emit("Subdivide: using the current unsaved in-memory "
                          "edit as input; earlier changes are kept.", "info")
        self.log.emit(f"Subdivide on {loaded.name}: 1 Catmull-Clark level "
                      "(quads kept); the result stays in memory until Save.",
                      "info")
        self.jobs.subdivide.start(loaded, levels=1,
                                  vertices=vertices, faces=faces)
        self.changed.emit()

    def run_close_holes(self):
        loaded = self._target("Close holes")
        if loaded is None:
            return
        vertices, faces = self.arrays_for(loaded)
        if vertices is not None:
            self.log.emit("Close holes: using the current unsaved in-memory "
                          "edit as input; earlier changes are kept.", "info")
        self.log.emit(f"Close holes on {loaded.name}: filling every boundary "
                      "hole (the rest of the mesh is untouched); the result "
                      "stays in memory until Save.", "info")
        self.jobs.close_holes.start(loaded, vertices=vertices, faces=faces)
        self.changed.emit()

    def run_delete_hidden(self):
        loaded = self._target("Delete hidden")
        if loaded is None:
            return
        viewer = self.viewer
        if not viewer.polygroups_hidden():
            return
        try:
            vertices, faces, keep_mask, kept, total, tri_groups = \
                viewer.polygroup_visible_geometry()
        except Exception as ex:  # noqa: BLE001 - report failures in the UI
            self.log.emit(f"Delete hidden: could not read the displayed "
                          f"geometry ({ex}).", "error")
            return
        if not len(faces) or not keep_mask.any():
            return
        self.log.emit(f"Delete hidden on {loaded.name}: removing "
                      f"{total - kept:,} of {total:,} polygon(s) "
                      f"({kept:,} stay); surviving polygroups keep their "
                      "colours and the result stays in memory until Bake "
                      "changes.", "info")
        import numpy as np

        face_groups = np.asarray(tri_groups, dtype=np.int64)[keep_mask]
        self.jobs.delete_faces.start(loaded, vertices=vertices, faces=faces,
                                     keep_mask=keep_mask,
                                     face_groups=face_groups)
        self.changed.emit()

    def run_mirror_merge(self, *, axes, keep_negative: bool):
        """Cut at the checked symmetry plane(s), mirror the kept side.

        The plane is the same world-space axis plane the brush symmetry
        uses (through the model's world origin); with several axes
        checked they chain (e.g. X then Z = quarter mirror).
        """
        loaded = self._target("Mirror and merge")
        if loaded is None:
            return
        axes = sorted({int(a) for a in axes})
        if not axes:
            self.log.emit("Mirror and merge: check at least one symmetry "
                          "axis (X / Y / Z) first.", "warn")
            return
        actors = self.viewer._split_actors()
        if not actors:
            self.log.emit("Mirror and merge: no displayed model.", "warn")
            return
        if len(actors) > 1:
            self.log.emit("Mirror and merge: the model has several parts; "
                          "the plane uses the first part's placement.",
                          "warn")
        import numpy as np

        matrix = actors[0].GetMatrix()
        if matrix is None:
            rotation = np.eye(3)
        else:
            rotation = np.array([[matrix.GetElement(r, c) for c in range(4)]
                                 for r in range(3)])[:, :3]
        planes = []
        for axis in axes:
            e = np.zeros(3)
            e[axis] = 1.0
            n_mesh = rotation.T @ e        # world axis in mesh space
            norm = float(np.linalg.norm(n_mesh))
            if norm < 1e-9:
                continue
            n_mesh /= norm
            # the pivot IS the actor's world origin, so the world plane
            # e_axis . w = pivot[axis] passes through the mesh-space
            # origin: with w = R p + t the plane is n_mesh . p = 0
            planes.append((n_mesh, 0.0))
        if keep_negative:
            # the tool keeps dist <= 0; flip every plane to keep +side
            planes = [(-n, -c) for n, c in planes]
        if not planes:
            self.log.emit("Mirror and merge: no usable symmetry plane.",
                          "warn")
            return
        vertices, faces = self.arrays_for(loaded)
        if vertices is not None:
            self.log.emit("Mirror and merge: using the current unsaved "
                          "in-memory edit as input; earlier changes are "
                          "kept.", "info")
        sides = " + ".join("XYZ"[a] for a in axes)
        self.log.emit(
            f"Mirror and merge on {loaded.name}: cutting at the {sides} "
            f"plane(s), keeping the "
            f"{'negative' if keep_negative else 'positive'} side and "
            "mirroring it; the result stays in memory until Bake changes.",
            "info")
        self.jobs.mirror_merge.start(loaded, planes=planes,
                                     vertices=vertices, faces=faces)
        self.changed.emit()

    def stop_running(self):
        """Ask the running edit operation to stop (cooperative)."""
        for name, ctrl in (("Reproject", self.jobs.reproject),
                           ("Polish", self.jobs.polish),
                           ("Decimate", self.jobs.decimate),
                           ("Subdivide", self.jobs.subdivide),
                           ("Close holes", self.jobs.close_holes),
                           ("Delete hidden", self.jobs.delete_faces),
                           ("Mirror and merge", self.jobs.mirror_merge),
                           ("Remesh", self.jobs.remesh)):
            if ctrl.running:
                ctrl.cancel()
                self.log.emit(f"{name}: stop requested; waiting for the "
                              "solver to reach its next cancellation point "
                              "...", "warn")

    # -- results / save ------------------------------------------------------
    def accept_result(self, label: str, result) -> bool:
        """Validate and display one finished edit operation."""
        try:
            source = Path(result["target_path"])
            vertices = result["vertices"]
            faces = result["faces"]
        except (KeyError, TypeError):
            self.log.emit(f"{label}: the worker returned no geometry.",
                          "error")
            self.changed.emit()
            return False

        current = self.viewer.current_path
        if current is None or current.resolve() != source.resolve():
            self.log.emit(f"{label}: the displayed model changed while the "
                          "run was in progress; its result was discarded.",
                          "warn")
            self.changed.emit()
            return False

        previous = self.pending
        history = []
        if previous is not None:
            history = list(previous.get("history")
                           or [previous.get("mode_label", "Edit")])
        history.append(label)
        result["history"] = history
        self.pending = result
        self._smooth_dirty = False
        self.changed.emit()

        try:
            shown = self.viewer.load_arrays(
                vertices, faces, preserve_camera=True,
                preserve_material=True, display_path=source)
        except Exception as ex:  # noqa: BLE001 - report in the UI
            self.log.emit(f"{label}: could not display the result in viewer: "
                          f"{ex}", "error")
            shown = False
        if shown:
            self.folder_changed.emit(source.parent)
            if len(history) > 1:
                self.log.emit(f"Edit stack: {' + '.join(history)}.", "info")
            self.log.emit(f"{label} geometry is in memory (not saved). Press "
                          f"Save to write {_edit_output_path(source).name}.",
                          "info")
            if "face_groups" in result:
                if self.viewer.restore_polygroups(result.get("face_groups")):
                    self.log.emit("Delete hidden: surviving polygroups "
                                  "restored with their original colours.",
                                  "info")
            return True

        self.pending = None
        self._smooth_bake_logged = False
        self._smooth_dirty = False
        self.changed.emit()
        self.log.emit(f"{label}: the result could not be displayed; nothing "
                      "was kept.", "error")
        return False

    def save(self, *, replace: bool) -> bool:
        """Write the pending stack as <model>_edit.glb or over the model."""
        if self.jobs.busy:
            return False
        self._materialize()   # lazy smooth-brush capture happens here
        label = "Bake" if replace else "Save"
        edit = self.pending
        if edit is None:
            self.log.emit(f"{label}: there is no unsaved edit to "
                          f"{'bake' if replace else 'save'}.", "warn")
            return False
        try:
            source = Path(edit["target_path"])
        except (KeyError, TypeError):
            self.log.emit(f"{label}: the unsaved edit is missing its "
                          "geometry.", "error")
            return False
        if replace and not source.is_file():
            self.log.emit(f"Bake: {source.name} no longer exists.", "warn")
            return False
        output = source if replace else _edit_output_path(source)
        return self._write(output, replace=replace)

    def _write(self, output: Path, *, replace: bool) -> bool:
        edit = self.pending
        if edit is None:
            return False
        try:
            source = Path(edit["target_path"])
            vertices = edit["vertices"]
            faces = edit.get("faces")
        except (KeyError, TypeError):
            self.log.emit("Save: the unsaved edit is missing its geometry.",
                          "error")
            return False
        history = list(edit.get("history") or [edit.get("mode_label", "Edit")])
        # capture polygroups of the displayed edit before the reload
        carry_faces = faces
        carry_groups = self.viewer.polygroups_face_groups()
        self.log.emit(f"{'Bake' if replace else 'Save'}: writing "
                      f"{' + '.join(history)} result to {output.name} ...",
                      "info")
        try:
            from tools._loader import load_tool

            stats = load_tool("reproject").save_reprojected(
                source, output, vertices, faces,
                log=lambda message, level="info":
                    self.log.emit(message, level))
        except Exception as ex:  # noqa: BLE001 - report in the UI
            self.failed.emit("Save failed", f"Save failed: {ex}")
            return False

        self.pending = None
        self._smooth_bake_logged = False
        self.changed.emit()
        if not stats.get("preserved_material", True):
            self.log.emit("Save: topology changed, so the original "
                          "UVs/material could not be carried over.", "warn")
        if self.viewer.load(output, preserve_camera=True,
                            preserve_material=True):
            self.saved.emit(output, True)
            self.log.emit(
                f"{'Baked changes into' if replace else 'Saved and showing'} "
                f"{output.name}.", "ok")
            if carry_groups is not None and carry_faces is not None:
                matched = self.viewer.restore_polygroups_by_faces(
                    vertices, carry_faces, carry_groups)
                if matched:
                    self.log.emit(f"Polygroups carried over: "
                                  f"{matched:,}/{len(carry_faces):,} faces "
                                  "matched their baked positions.", "ok")
                else:
                    self.log.emit("The polygroups could not be carried over "
                                  "(the written topology does not match the "
                                  "edited mesh).", "warn")
        else:
            self.log.emit(f"Saved {output.name}, but the viewer could not "
                          "display it.", "warn")
            self.saved.emit(output, False)
        # Publish the saved OBJ next to the GLB, like every other result.
        self.jobs.convert(output)
        return True

    # -- split and save ------------------------------------------------------
    def save_split(self):
        """Show every polygroup, name it, then write one GLB per group."""
        if self.jobs.busy:
            return
        viewer = self.viewer
        loaded = viewer.current_path
        if loaded is None or not loaded.is_file():
            self.log.emit("Split: no model is loaded.", "warn")
            return
        if self.pending is not None:
            self.log.emit("Split: save or discard the unsaved edit first.",
                          "warn")
            return
        if not viewer.polygroup_count:
            self.log.emit("Split: no painted polygroups. Select faces and "
                          "press Ctrl+W first.", "warn")
            return

        previous_preset = viewer.preset
        previous_visibility = viewer.polygroup_visibility()
        prepared = viewer.prepare_polygroup_split()
        try:
            geometry = viewer.polygroup_split_geometry()
            groups = sorted(geometry)
            painted = [group for group in groups
                       if group != viewer_mod.DEFAULT_POLYGROUP]
            if not painted:
                self.log.emit("Split: no painted polygroups to export.",
                              "warn")
                return
            names = self._ask_split_names(groups, loaded.stem)
            if names is None:
                self.log.emit("Split: cancelled; nothing was written.",
                              "warn")
                return
            outputs = [
                (group, loaded.with_name(f"{loaded.stem}_{suffix}.glb"))
                for group, suffix in names
            ]
            existing = [path.name for _, path in outputs if path.exists()]
            if existing:
                answer = QMessageBox.question(
                    self.parent(), "Split and save",
                    "These files already exist and will be overwritten:\n\n"
                    + "\n".join(existing)
                    + "\n\nContinue?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if answer != QMessageBox.Yes:
                    self.log.emit("Split: cancelled; existing files were left "
                                  "untouched.", "warn")
                    return
            import trimesh

            written = []
            for group, output in outputs:
                vertices, faces = geometry[group]
                self.log.emit(f"Split: writing {output.name} "
                              f"({len(faces):,} faces) ...", "info")
                try:
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces,
                                           process=False, validate=False)
                    mesh.export(str(output))
                except Exception as ex:  # noqa: BLE001 - report in the UI
                    self.log.emit(f"Split: {output.name} failed: {ex}",
                                  "error")
                    QMessageBox.warning(self.parent(), "Split failed", str(ex))
                    break
                written.append(output)
            if written:
                self.folder_changed.emit(loaded.parent)
            if len(written) == len(outputs):
                self.log.emit(f"Split: wrote {len(written)} GLB file(s) next "
                              f"to {loaded.name}.", "ok")
            elif written:
                self.log.emit(f"Split: {len(written)} of {len(outputs)} files "
                              "were written before the failure.", "warn")
        finally:
            if prepared:
                viewer.set_material_preset(previous_preset)
            viewer.set_polygroup_visibility(previous_visibility)
            viewer.render_now()

    def _ask_split_names(self, groups, stem):
        """Solo each group and collect its suffix; None when cancelled."""
        names = []
        used = {}
        for index, group in enumerate(groups, 1):
            self.viewer.solo_polygroup(group)
            self.viewer.render_now()
            QApplication.processEvents()
            default = f"group{group}"
            while True:
                dialog = SplitNameDialog(group, index, len(groups),
                                          default, stem, self.parent())
                if dialog.exec() != QDialog.Accepted:
                    return None
                suffix = dialog.suffix()
                if not suffix:
                    continue
                key = suffix.casefold()
                if key in used:
                    QMessageBox.warning(
                        self.parent(), "Split and save",
                        f'Suffix "{suffix}" is already used for group '
                        f"{used[key]}. Choose a different suffix.")
                    continue
                used[key] = group
                names.append((group, suffix))
                break
        return names
