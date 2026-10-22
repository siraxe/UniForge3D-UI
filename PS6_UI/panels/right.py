"""Right edit panel: reproject, edit mesh, remesh, save footer."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame,
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea, QSpinBox,
    QVBoxLayout, QWidget,
)

from utils.layout import accordion_section
import hotkeys

class RightPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        panel = self
        panel.setObjectName("editPanel")
        panel.setMinimumWidth(250)
        panel.setMaximumWidth(560)
        panel.setStyleSheet(
            "QWidget#editPanel { background: #232529;"
            " border-left: 1px solid #3a3d42; }")
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollBar:vertical { background: #1d1f22; width: 10px; }"
            "QScrollBar::handle:vertical { background: #4a4e55;"
            " min-height: 24px; border-radius: 4px; }")
        content = QWidget()
        content.setObjectName("editPanelContent")
        content.setStyleSheet(
            "QWidget#editPanelContent { background: #232529; }")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        root = layout            # panel-level layout: separators + accordions

        # -- Reproject ------------------------------------------------------
        layout = accordion_section(root, "Reproject:", first=True, expanded=False)

        self.reproject_mode = QComboBox()
        # data values mirror reproject.MODE_* (that module is imported
        # lazily in the worker so app startup stays light).
        for label, mode in (
            ("ProjectAll snap", "project"),
            ("Normal only", "normal"),
            ("High-pass detail", "highpass"),
            ("Higher subdiv", "higher_subdiv"),
        ):
            self.reproject_mode.addItem(label, mode)
        self.reproject_mode.setCurrentIndex(
            self.reproject_mode.findData("highpass"))
        self.reproject_mode.setToolTip("Reprojection detail mode")
        layout.addWidget(self.reproject_mode)

        self.reproject_hint = QLabel("")
        self.reproject_hint.setWordWrap(True)
        self.reproject_hint.setStyleSheet("color: #9a9ea6; font-size: 10px;")
        layout.addWidget(self.reproject_hint)

        self.reproject_dist = QDoubleSpinBox()
        self.reproject_dist.setDecimals(3)
        self.reproject_dist.setRange(0.001, 1.0)
        self.reproject_dist.setSingleStep(0.005)
        self.reproject_dist.setValue(0.030)
        self.reproject_dist.setMinimumWidth(100)
        self.reproject_dist.setToolTip("Projection distance threshold")

        self.reproject_strength = QSpinBox()
        self.reproject_strength.setRange(1, 100)
        self.reproject_strength.setValue(100)
        self.reproject_strength.setSuffix("%")
        self.reproject_strength.setMinimumWidth(100)
        self.reproject_strength.setToolTip("Projection strength")

        self.reproject_blur = QSpinBox()
        self.reproject_blur.setRange(0, 10)
        self.reproject_blur.setValue(0)
        self.reproject_blur.setMinimumWidth(100)
        self.reproject_blur.setToolTip("Pre-projection smoothing")

        self.reproject_shell = QDoubleSpinBox()
        self.reproject_shell.setDecimals(3)
        self.reproject_shell.setRange(-0.5, 0.5)
        self.reproject_shell.setSingleStep(0.005)
        self.reproject_shell.setValue(0.0)
        self.reproject_shell.setMinimumWidth(100)
        self.reproject_shell.setToolTip("Surface offset before projecting")

        self.reproject_detail = QSpinBox()
        self.reproject_detail.setRange(1, 50)
        self.reproject_detail.setValue(5)
        self.reproject_detail.setMinimumWidth(100)
        self.reproject_detail.setToolTip("High-pass detail separation")

        self.reproject_subdiv = QSpinBox()
        self.reproject_subdiv.setRange(1, 3)
        self.reproject_subdiv.setValue(1)
        self.reproject_subdiv.setMinimumWidth(100)
        self.reproject_subdiv.setToolTip("Pre-snap subdivision levels")

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(3)
        form.addRow("dist", self.reproject_dist)
        form.addRow("strength", self.reproject_strength)
        form.addRow("blur", self.reproject_blur)
        form.addRow("shell", self.reproject_shell)
        form.addRow("detail", self.reproject_detail)
        form.addRow("subdiv", self.reproject_subdiv)
        layout.addLayout(form)

        self.btn_reproject = QPushButton("Reproject")
        self.btn_reproject.setFixedHeight(28)
        self.btn_reproject.setToolTip("Reproject donor surface detail")
        self.btn_reproject.clicked.connect(self._run_reproject)
        layout.addWidget(self.btn_reproject)

        # -- Edit mesh (Polish | Decimate) -----------------------------------
        layout = accordion_section(root, "Edit mesh:", expanded=True)

        self.polish_amount = QSpinBox()
        self.polish_amount.setRange(0, 100)
        self.polish_amount.setValue(2)
        self.polish_amount.setSuffix("%")
        self.polish_amount.setMinimumWidth(100)
        self.polish_amount.setToolTip("Polish strength")

        self.polish_peaks = QCheckBox("peaks")
        self.polish_peaks.setChecked(True)
        self.polish_peaks.setToolTip("Alternate polish path")

        self.btn_polish = QPushButton("Polish")
        self.btn_polish.setFixedHeight(28)
        self.btn_polish.setToolTip("Relax the displayed mesh")
        self.btn_polish.clicked.connect(self._run_polish)

        self.decimate_tris = QSpinBox()
        self.decimate_tris.setRange(1, 2000)
        self.decimate_tris.setValue(150)
        self.decimate_tris.setSuffix("k")
        self.decimate_tris.setMinimumWidth(100)
        self.decimate_tris.setToolTip("Target triangle count (thousands)")

        self.btn_decimate = QPushButton("Decimate")
        self.btn_decimate.setFixedHeight(28)
        self.btn_decimate.setToolTip("Decimate to target triangles")
        self.btn_decimate.clicked.connect(self._run_decimate)

        self.btn_subdivide = QPushButton("Subdivide")
        self.btn_subdivide.setFixedHeight(28)
        self.btn_subdivide.setToolTip("One Catmull-Clark subdivision level")
        self.btn_subdivide.clicked.connect(self._run_subdivide)

        self.btn_close_holes = QPushButton("Close holes")
        self.btn_close_holes.setFixedHeight(28)
        self.btn_close_holes.setToolTip("Close boundary holes")
        self.btn_close_holes.clicked.connect(self._run_close_holes)

        self.btn_delete_hidden = QPushButton("Delete hidden")
        self.btn_delete_hidden.setFixedHeight(28)
        self.btn_delete_hidden.setToolTip("Delete hidden polygroup faces")
        self.btn_delete_hidden.clicked.connect(self._run_delete_hidden)

        self.sym_opposite = QCheckBox()
        self.sym_opposite.setToolTip(
            "MirrorAndMerge side switch: unchecked keeps the POSITIVE side "
            "of the plane and deletes the negative; checked keeps the "
            "NEGATIVE side instead (uses the X/Y/Z axis checkboxes)")
        self.btn_mirror_merge = QPushButton("MirrorAndMerge")
        self.btn_mirror_merge.setFixedHeight(28)
        self.btn_mirror_merge.setToolTip(
            "Cut the mesh with the checked symmetry plane(s), delete the "
            "other side, mirror the kept side and weld the halves "
            "together (needs at least one X/Y/Z axis checked)")
        self.btn_mirror_merge.clicked.connect(self._run_mirror_merge)

        columns = QHBoxLayout()
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(10)
        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(6)
        left.addWidget(self.polish_amount)
        polish_row = QHBoxLayout()
        polish_row.setContentsMargins(0, 0, 0, 0)
        polish_row.setSpacing(6)
        polish_row.addWidget(self.polish_peaks)
        polish_row.addWidget(self.btn_polish, 1)
        left.addLayout(polish_row)
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(6)
        right.addWidget(self.decimate_tris)
        right.addWidget(self.btn_decimate)
        columns.addLayout(left, 1)
        columns.addLayout(right, 1)
        layout.addLayout(columns)
        subdivide_row = QHBoxLayout()
        subdivide_row.setContentsMargins(0, 0, 0, 0)
        subdivide_row.setSpacing(6)
        subdivide_row.addWidget(self.btn_subdivide, 1)
        subdivide_row.addWidget(self.btn_close_holes, 1)
        layout.addLayout(subdivide_row)
        delete_row = QHBoxLayout()
        delete_row.setContentsMargins(0, 0, 0, 0)
        delete_row.setSpacing(6)
        delete_row.addWidget(self.sym_opposite)
        delete_row.addWidget(self.btn_mirror_merge, 1)
        delete_row.addWidget(self.btn_delete_hidden, 1)
        layout.addLayout(delete_row)

        # -- Symmetry (Draw-mode brush mirroring) -----------------------------
        sym_row = QHBoxLayout()
        sym_row.setContentsMargins(0, 0, 0, 0)
        sym_row.setSpacing(6)
        sym_row.addWidget(QLabel("Symmetry:"))
        self.symmetry_enable = QCheckBox()
        self.symmetry_enable.setToolTip(
            "Mirror Draw-mode brush edits (move / smooth) to the mirrored "
            "side of the checked axis planes (hotkey: "
            + hotkeys.seq("symmetry_toggle") + ")")
        sym_row.addWidget(self.symmetry_enable)
        sym_row.addStretch(1)
        self.btn_resym = QPushButton("reSym")
        self.btn_resym.setFixedHeight(24)
        self.btn_resym.setToolTip(
            "Align the model's symmetry plane with the checked axis - yaw "
            "only, around the model's own UP: click two opposite feature "
            "tips (red), two points along the body (dark blue), then "
            "three points on the model's bottom (dark green, floor), "
            "confirm, and Bake changes. Esc cancels; left-drag orbits, "
            "right-drag zooms while placing points.")
        self.btn_resym.clicked.connect(self._start_resym)
        sym_row.addWidget(self.btn_resym)
        layout.addLayout(sym_row)

        self.sym_x = QCheckBox("X")
        self.sym_x.setChecked(True)
        self.sym_y = QCheckBox("Y")
        self.sym_z = QCheckBox("Z")
        for checkbox in (self.sym_x, self.sym_y, self.sym_z):
            checkbox.setToolTip(
                "Mirror axis for Draw-mode brush edits: a world-space plane "
                "through the model's world origin (used while the Symmetry "
                "switch is on)")
        axes_row = QHBoxLayout()
        axes_row.setContentsMargins(0, 0, 0, 0)
        axes_row.setSpacing(6)
        axes_row.addWidget(self.sym_x, 1)
        axes_row.addWidget(self.sym_y, 1)
        axes_row.addWidget(self.sym_z, 1)
        layout.addLayout(axes_row)

        # -- Remesh ---------------------------------------------------------
        layout = accordion_section(root, "Remesh")

        self.remesh_quads = QSpinBox()
        self.remesh_quads.setRange(0, 150000)
        self.remesh_quads.setValue(0)
        self.remesh_quads.setSpecialValueText("auto")
        self.remesh_quads.setMinimumWidth(100)
        self.remesh_quads.setToolTip("Target quad count")

        self.remesh_sym = QComboBox()
        for label, axes in (
            ("none", ""), ("X", "X"), ("Y", "Y"), ("Z", "Z"),
            ("XY", "XY"), ("XZ", "XZ"), ("YZ", "YZ"), ("XYZ", "XYZ"),
        ):
            self.remesh_sym.addItem(label, axes)
        self.remesh_sym.setToolTip("Symmetry axes (unused)")

        self.remesh_density = QDoubleSpinBox()
        self.remesh_density.setDecimals(2)
        self.remesh_density.setRange(0.25, 4.0)
        self.remesh_density.setSingleStep(0.25)
        self.remesh_density.setValue(1.0)
        self.remesh_density.setMinimumWidth(100)
        self.remesh_density.setToolTip("Quad density multiplier")

        self.remesh_freeze = QCheckBox("freeze borders")
        self.remesh_freeze.setToolTip("Freeze boundary edges (unused)")

        remesh_form = QFormLayout()
        remesh_form.setContentsMargins(0, 0, 0, 0)
        remesh_form.setHorizontalSpacing(8)
        remesh_form.setVerticalSpacing(3)
        remesh_form.addRow("quads", self.remesh_quads)
        remesh_form.addRow("symmetry", self.remesh_sym)
        remesh_form.addRow("density", self.remesh_density)
        remesh_form.addRow(self.remesh_freeze)
        layout.addLayout(remesh_form)

        self.remesh_status = QLabel("")
        self.remesh_status.setWordWrap(True)
        self.remesh_status.setStyleSheet(
            "color: #9a9ea6; font-size: 10px;")
        self.remesh_status.setToolTip("Native remesher status")
        layout.addWidget(self.remesh_status)

        self.btn_remesh = QPushButton("Remesh")
        self.btn_remesh.setFixedHeight(28)
        self.btn_remesh.setToolTip("Run quad remesher")
        self.btn_remesh.clicked.connect(self._run_remesh)
        layout.addWidget(self.btn_remesh)

        # -- Remesh_LATO2 (learned retopology) ------------------------------
        layout = accordion_section(root, "Remesh_LATO2:")

        self.lato2_verts = QSpinBox()
        self.lato2_verts.setRange(200, 5000)
        self.lato2_verts.setSingleStep(100)
        self.lato2_verts.setValue(2000)
        self.lato2_verts.setMinimumWidth(100)
        self.lato2_verts.setToolTip("Target vertex count")

        self.lato2_fill_rings = QCheckBox("fill quad rings")
        self.lato2_fill_rings.setChecked(False)
        self.lato2_fill_rings.setToolTip(
            "Off by default: fewer non-manifold edges, but more holes left "
            "for Close Holes. On matches LATO.2's repo default and can fill "
            "some holes, but adds overlapping quad-ring triangles.")
        self.lato2_drop_tiny = QCheckBox("drop small patches")
        self.lato2_drop_tiny.setChecked(True)
        self.lato2_drop_tiny.setToolTip(
            "Remove tiny disconnected surface islands left by the edge "
            "decoder; they render as overlapping slivers. Remaining holes "
            "can be repaired with Close Holes.")

        lato2_form = QFormLayout()
        lato2_form.setContentsMargins(0, 0, 0, 0)
        lato2_form.setHorizontalSpacing(8)
        lato2_form.setVerticalSpacing(3)
        lato2_form.addRow("verts", self.lato2_verts)
        lato2_form.addRow(self.lato2_fill_rings)
        lato2_form.addRow(self.lato2_drop_tiny)
        layout.addLayout(lato2_form)

        self.lato2_hint = QLabel(
            "V-Flow + T-Flow retopology: regenerates vertices and "
            "connectivity from the displayed mesh.")
        self.lato2_hint.setWordWrap(True)
        self.lato2_hint.setStyleSheet("color: #9a9ea6; font-size: 10px;")
        layout.addWidget(self.lato2_hint)

        self.btn_lato2 = QPushButton("Run LATO.2")
        self.btn_lato2.setFixedHeight(28)
        self.btn_lato2.setToolTip("Retopologize the displayed mesh")
        self.btn_lato2.clicked.connect(lambda: self.win.pipeline.run_lato2())
        layout.addWidget(self.btn_lato2)

        root.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        # -- pinned save footer ---------------------------------------------
        footer = QWidget()
        footer.setObjectName("editFooter")
        footer.setStyleSheet(
            "QWidget#editFooter { background: #232529;"
            " border-top: 1px solid #3a3d42; }")
        foot = QVBoxLayout(footer)
        foot.setContentsMargins(10, 8, 10, 10)
        foot.setSpacing(6)
        self.edit_state = QLabel("No unsaved edits")
        self.edit_state.setWordWrap(True)
        self.edit_state.setStyleSheet(
            "color: #888c92; font-size: 10px;")
        foot.addWidget(self.edit_state)

        self.btn_stop_edit = QPushButton("Stop")
        self.btn_stop_edit.setFixedHeight(26)
        self.btn_stop_edit.setStyleSheet(
            "QPushButton { background-color: #a04040; color: white; "
            "font-weight: 600; }"
            "QPushButton:disabled { background-color: #555555; "
            "color: #aaaaaa; }")
        self.btn_stop_edit.setToolTip("Stop the running edit")
        self.btn_stop_edit.clicked.connect(
            lambda: self.win.edits.stop_running())
        self.btn_stop_edit.setVisible(False)
        foot.addWidget(self.btn_stop_edit)

        self.btn_save = QPushButton("Save edit")
        self.btn_save.setFixedHeight(30)
        self.btn_save.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; "
            "font-weight: 600; }"
            "QPushButton:disabled { background-color: #555555; "
            "color: #aaaaaa; }")
        self.btn_save.setToolTip("Save edit as _edit.glb")
        self.btn_save.clicked.connect(self._save_edit)

        self.btn_save_replace = QPushButton("Bake changes")
        self.btn_save_replace.setFixedHeight(30)
        self.btn_save_replace.setStyleSheet(
            "QPushButton { background-color: #b3661f; color: white; "
            "font-weight: 600; }"
            "QPushButton:disabled { background-color: #555555; "
            "color: #aaaaaa; }")
        self.btn_save_replace.setToolTip("Overwrite model with edit")
        self.btn_save_replace.clicked.connect(self._bake_edit)

        save_row = QHBoxLayout()
        save_row.setContentsMargins(0, 0, 0, 0)
        save_row.setSpacing(6)
        save_row.addWidget(self.btn_save, 1)
        save_row.addWidget(self.btn_save_replace, 1)
        foot.addLayout(save_row)

        self.btn_split = QPushButton("Split and save")
        self.btn_split.setFixedHeight(30)
        self.btn_split.setStyleSheet(
            "QPushButton { background-color: #2f5d8a; color: white; "
            "font-weight: 600; }"
            "QPushButton:disabled { background-color: #555555; "
            "color: #aaaaaa; }")
        self.btn_split.setToolTip("Export polygroups as GLBs")
        self.btn_split.clicked.connect(lambda: self.win.edits.save_split())
        foot.addWidget(self.btn_split)
        outer.addWidget(footer)

        self.reproject_mode.currentIndexChanged.connect(
            self._on_reproject_mode_changed)
        self._update_reproject_hint()
        self.refresh_remesh_status()

    def set_state(self, state):
        edit_on = state.idle and state.has_model
        file_op_on = edit_on and not state.pending_edit
        self.btn_reproject.setEnabled(edit_on and state.donor_ok)
        self.btn_polish.setEnabled(edit_on)
        self.btn_stop_edit.setVisible(state.edit_running)
        self.btn_decimate.setEnabled(edit_on)
        self.decimate_tris.setEnabled(edit_on)
        self.btn_subdivide.setEnabled(edit_on)
        self.btn_close_holes.setEnabled(edit_on)
        self.btn_delete_hidden.setEnabled(edit_on)
        self.btn_mirror_merge.setEnabled(edit_on)
        self.sym_opposite.setEnabled(edit_on)
        self.refresh_remesh_status()
        self.btn_remesh.setEnabled(edit_on and self._remesh_ready)
        self.lato2_verts.setEnabled(file_op_on)
        self.lato2_fill_rings.setEnabled(file_op_on)
        self.lato2_drop_tiny.setEnabled(file_op_on)
        self.btn_lato2.setEnabled(file_op_on)
        self.btn_save.setEnabled(state.idle and state.pending_edit)
        self.btn_save_replace.setEnabled(state.idle and state.pending_edit)
        self.btn_split.setEnabled(state.idle and state.has_model
                                  and not state.pending_edit)

        mode = self.reproject_mode.currentData()
        self.reproject_mode.setEnabled(edit_on)
        self.reproject_strength.setEnabled(edit_on)
        self.reproject_blur.setEnabled(edit_on)
        self.reproject_shell.setEnabled(edit_on)
        self.reproject_dist.setEnabled(edit_on and mode != "higher_subdiv")
        self.reproject_detail.setEnabled(edit_on and mode == "highpass")
        self.reproject_subdiv.setEnabled(edit_on
                                         and mode == "higher_subdiv")
        for widget in (self.polish_amount, self.polish_peaks,
                       self.remesh_quads, self.remesh_sym,
                       self.remesh_density, self.remesh_freeze):
            widget.setEnabled(edit_on)

    def refresh_edit_state(self):
        edit = self.win.edits.pending
        if edit is None:
            self.edit_state.setText("No unsaved edits")
            self.edit_state.setStyleSheet(
                "color: #888c92; font-size: 10px;")
            return
        count = int(edit.get("vertex_count", 0))
        history = list(edit.get("history") or [edit.get("mode_label", "Edit")])
        if len(history) > 2:
            label = " + ".join(history[-2:]) + f" (+{len(history) - 2})"
        else:
            label = " + ".join(history)
        self.edit_state.setText(
            f"Unsaved edit: {label}\n{count:,} vertices")
        self.edit_state.setStyleSheet(
            "color: #f0c674; font-size: 10px;")

    def _run_reproject(self):
        self.win.edits.run_reproject(
            mode=self.reproject_mode.currentData(),
            mode_label=self.reproject_mode.currentText(),
            dist=self.reproject_dist.value(),
            strength=self.reproject_strength.value() / 100.0,
            blur=self.reproject_blur.value(),
            shell=self.reproject_shell.value(),
            detail=self.reproject_detail.value(),
            subdiv=self.reproject_subdiv.value())

    def _run_polish(self):
        self.win.edits.run_polish(amount=self.polish_amount.value(),
                                  preserve_peaks=self.polish_peaks.isChecked())

    def _run_remesh(self):
        self.refresh_remesh_status()
        if not self._remesh_ready:
            message = ("Remesh: the compiled remesher is unavailable "
                       "(tools/remesh_rust/remesh_rust.dll).")
            self.win.log(message, "warn")
            QMessageBox.information(self, "Remesh unavailable", message)
            return
        self.win.edits.run_remesh(
            target_quads=self.remesh_quads.value(),
            symmetry=self.remesh_sym.currentData() or "",
            freeze_borders=self.remesh_freeze.isChecked(),
            var_density_ratio=self.remesh_density.value())

    def _run_decimate(self):
        self.win.edits.run_decimate(
            target_tris=self.decimate_tris.value() * 1000)

    def _run_subdivide(self):
        self.win.edits.run_subdivide()

    def _run_close_holes(self):
        self.win.edits.run_close_holes()

    def _run_delete_hidden(self):
        self.win.edits.run_delete_hidden()

    def _run_mirror_merge(self):
        axes = [axis for axis, checkbox in
                enumerate((self.sym_x, self.sym_y, self.sym_z))
                if checkbox.isChecked()]
        self.win.edits.run_mirror_merge(
            axes=axes, keep_negative=self.sym_opposite.isChecked())

    def _start_resym(self):
        self.win.viewer.resym.start()

    def _save_edit(self):
        self.win.edits.save(replace=False)

    def _bake_edit(self):
        self.win.edits.save(replace=True)

    def _on_reproject_mode_changed(self):
        self._update_reproject_hint()
        self.win._refresh_buttons()

    def _update_reproject_hint(self):
        hints = {
            "project":
                "Exact ProjectAll snap: vertices within Dist move onto the "
                "ghosted donor's surface; connectivity is kept.",
            "normal":
                "Moves only along the target normal, so the repaired layout "
                "does not slide sideways.",
            "highpass":
                "Copies only the donor's fine detail (donor minus its "
                "smoothed shape); the repaired silhouette is kept.",
            "higher_subdiv":
                "Subdivides the target first, then snaps every vertex onto "
                "the donor. Dist is ignored; the output topology changes.",
        }
        mode = self.reproject_mode.currentData()
        self.reproject_hint.setText(hints.get(mode, ""))

    def on_remesh_progress(self, percent: int, stage: str):
        """Live progress line while the remesher pipeline runs."""
        self.remesh_status.setText(
            f"Remesh: {stage} ... {int(percent)}%")
        self.remesh_status.setStyleSheet(
            "color: #f0c674; font-size: 10px;")

    def refresh_remesh_status(self):
        """Ready only when the compiled Rust remesher loads."""
        if self.win.jobs.remesh.running:
            return
        try:
            import importlib.util

            root = Path(__file__).resolve().parents[2]
            bridge = root / "tools" / "remesh_rust" / "bridge.py"
            rust_ready = False
            if bridge.is_file():
                spec = importlib.util.spec_from_file_location(
                    "remesh_rust_bridge_probe", bridge)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                rust_ready = bool(mod.remesh_available())
        except Exception:
            rust_ready = False
        self._remesh_ready = rust_ready
        if not rust_ready:
            self.remesh_status.setText(
                "Remesh unavailable: tools/remesh_rust/remesh_rust.dll "
                "missing or not loadable (cargo build --release && "
                "cp target/release/remesh_rust.dll .)")
            self.remesh_status.setStyleSheet(
                "color: #e06c75; font-size: 10px;")
        if self._remesh_ready:
            # no "Ready" line by default; the label only appears when the
            # remesher runs (progress) or is unavailable (error)
            self.remesh_status.setText("")
        else:
            self.remesh_status.setText(
                f"Remesh unavailable: {state.get('error', 'unknown')}")
            self.remesh_status.setStyleSheet(
                "color: #e06c75; font-size: 10px;")
