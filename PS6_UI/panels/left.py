"""Left controls column: images, generation/upscale settings, log."""

from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from utils.layout import underlined_label
from widgets import THUMB_SIZE, _Thumb

class LeftPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.setMinimumWidth(260)   # keep labels usable when dragged narrow
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Input images with thumbnails ------------------------------------
        grp_input = QGroupBox("Input image(s)")
        v = QVBoxLayout(grp_input)
        self.thumb_list = QListWidget()
        self.thumb_list.setViewMode(QListWidget.IconMode)
        self.thumb_list.setResizeMode(QListWidget.Adjust)
        self.thumb_list.setMovement(QListWidget.Static)
        self.thumb_list.setSpacing(6)
        self.thumb_list.setUniformItemSizes(True)
        self.thumb_list.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self.thumb_list.setMaximumHeight(THUMB_SIZE * 2 + 24)
        self.thumb_list.setStyleSheet(
            "QListWidget::item { border: none; background: transparent; }"
            "QListWidget::item:selected { border: none; background: transparent; }")
        self.thumb_list.itemClicked.connect(self._on_thumbnail_item_clicked)
        v.addWidget(self.thumb_list)
        row = QHBoxLayout()
        b_add = QPushButton("Add...")
        b_add.clicked.connect(self.win.on_add_images)
        b_clear = QPushButton("Clear")
        b_clear.clicked.connect(self.win.on_clear_images)
        self.btn_rmbg = QPushButton("RMBG")
        self.btn_rmbg.setToolTip("Remove image backgrounds")
        self.btn_rmbg.clicked.connect(
            lambda: self.win.pipeline.remove_background())
        # Allow the row (and therefore the whole left panel) to get narrow
        # without the buttons pinning it at their natural width.
        for b in (b_add, b_clear, self.btn_rmbg):
            b.setMinimumWidth(50)
        row.addWidget(b_add)
        row.addWidget(b_clear)
        row.addWidget(self.btn_rmbg)
        row.addStretch(1)
        v.addLayout(row)
        self.cutout_label = QLabel("")
        self.cutout_label.setStyleSheet("color: #7bd88f; font-size: 10px;")
        self.cutout_label.setVisible(False)
        v.addWidget(self.cutout_label)
        self.cutout_list = QListWidget()
        self.cutout_list.setViewMode(QListWidget.IconMode)
        self.cutout_list.setResizeMode(QListWidget.Adjust)
        self.cutout_list.setMovement(QListWidget.Static)
        self.cutout_list.setSpacing(6)
        self.cutout_list.setUniformItemSizes(True)
        self.cutout_list.setMaximumHeight(THUMB_SIZE + 12)
        self.cutout_list.setStyleSheet(
            "QListWidget { border: none; background: transparent; }")
        self.cutout_list.setVisible(False)
        v.addWidget(self.cutout_list)
        layout.addWidget(grp_input)

        # UniForge3D settings ---------------------------------------------
        grp_set = QGroupBox("UniForge3D settings")
        form = QFormLayout(grp_set)
        # When the controls column is dragged narrow, labels stack above
        # their fields instead of forcing a wide two-column minimum.
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Textured (PBR)", "Geometry only"])
        form.addRow("Output mode", self.mode_combo)

        self.grid_combo = QComboBox()
        self.grid_combo.addItems(["256", "384", "512", "768", "1024"])
        self.grid_combo.setCurrentText("1024")
        form.addRow("Geometry grid", self.grid_combo)

        self.tex_combo = QComboBox()
        self.tex_combo.addItems(["1024", "2048", "4096"])
        self.tex_combo.setCurrentText("2048")
        self.tex_combo.setToolTip("PBR texture resolution")
        form.addRow("Texture size", self.tex_combo)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(0, 64)
        self.steps_spin.setValue(20)   # pipeline default (all denoise stages)
        self.steps_spin.setSpecialValueText("default")
        form.addRow("Sampler steps", self.steps_spin)

        self.low_vram_check = QCheckBox("Low VRAM")
        self.low_vram_check.setChecked(True)
        self.low_vram_check.setToolTip("Low VRAM mode")
        self.low_vram_check.toggled.connect(
            lambda checked: self.win.pipeline.set_low_vram(checked))
        form.addRow("Memory mode", self.low_vram_check)

        self.stop_stage1_check = QCheckBox("Stop at 1st stage")
        self.stop_stage1_check.setChecked(True)
        self.stop_stage1_check.setToolTip("Stop after Shape-512")
        self._stop_stage1_previous_mode = self.mode_combo.currentIndex()
        self.stop_stage1_check.toggled.connect(self._on_stop_stage1_toggled)
        form.addRow("Early stop", self.stop_stage1_check)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 2_147_483_647)
        self.seed_spin.setValue(-1)
        self.seed_spin.setMinimumWidth(80)
        self.seed_spin.setToolTip("Random seed per run")
        # Toggles the field between "random" and the seed used by the last
        # completed run; the label follows the field's current mode.
        self.btn_seed = QPushButton("Use last")
        self.btn_seed.setMinimumWidth(60)
        self.btn_seed.clicked.connect(self.on_seed_button)
        self.seed_spin.valueChanged.connect(self._update_seed_button)
        seed_row = QHBoxLayout()
        seed_row.setContentsMargins(0, 0, 0, 0)
        seed_row.addWidget(self.seed_spin, 1)
        seed_row.addWidget(self.btn_seed)
        seed_box = QWidget()
        seed_box.setLayout(seed_row)
        form.addRow("Seed", seed_box)
        self._update_seed_button()

        # Geometry only is the default. Texture size stays editable because
        # Generate Texture uses it to re-texture an existing geometry result.
        self.mode_combo.setCurrentIndex(1)

        run_row = QHBoxLayout()
        run_row.setContentsMargins(0, 4, 0, 0)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setFixedHeight(32)
        self.btn_stop.setMinimumWidth(55)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(lambda: self.win.pipeline.stop())
        self.btn_generate = QPushButton("Generate")
        self.btn_generate.setFixedHeight(32)
        self.btn_generate.setMinimumWidth(70)
        self.btn_generate.clicked.connect(lambda: self.win.pipeline.generate())
        # The main action: solid green, white text; gray when disabled.
        self.btn_generate.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; "
            "font-weight: 600; }"
            "QPushButton:disabled { background-color: #555555; "
            "color: #aaaaaa; }")
        self.btn_generate_texture = QPushButton("Generate Texture")
        self.btn_generate_texture.setFixedHeight(32)
        self.btn_generate_texture.setToolTip("Texture from geometry-only result")
        self.btn_generate_texture.setVisible(False)
        self.btn_generate_texture.clicked.connect(
            lambda: self.win.pipeline.generate_texture())
        # Swapped: Stop sits on the left, Generate on the right.
        run_row.addWidget(self.btn_stop)
        run_row.addWidget(self.btn_generate_texture)
        run_row.addStretch(1)
        run_row.addWidget(self.btn_generate)
        run_box = QWidget()
        run_box.setLayout(run_row)
        form.addRow(run_box)

        layout.addWidget(grp_set)

        # Upscale settings --------------------------------------------------
        # Visible from the start; enabled once a model is displayed. The
        # upscale/repair/refine buttons all act on that model.
        self.grp_upscale = QGroupBox("Upscale settings")
        upscale_col = QVBoxLayout(self.grp_upscale)
        upscale_col.setContentsMargins(8, 6, 8, 6)

        # Top row: the scene-mesh switch for the 2nd stage.
        self.upscale_scene = QCheckBox("Apply to mesh in scene")
        self.upscale_scene.setChecked(False)
        self.upscale_scene.setToolTip("Upscale the scene mesh")
        upscale_col.addWidget(self.upscale_scene)

        # CelloCut row (Run 2nd stage shares it, offset a bit to the right).
        cello_row = QHBoxLayout()
        cello_row.setContentsMargins(0, 0, 0, 0)
        self.btn_cellocut = QPushButton("Run CelloCut (watertight)")
        self.btn_cellocut.setFixedHeight(28)
        self.btn_cellocut.setMinimumWidth(90)
        self.btn_cellocut.setToolTip("Repair mesh with CelloCut")
        self.btn_cellocut.clicked.connect(
            lambda: self.win.pipeline.run_cellocut())
        cello_row.addWidget(self.btn_cellocut)

        # "Run 2nd stage" shares the CelloCut row, offset a bit to the right.
        self.btn_stage2 = QPushButton("Run 2nd stage")
        self.btn_stage2.setFixedHeight(28)
        self.btn_stage2.setMinimumWidth(70)
        # Slightly green text on the default button.
        self.btn_stage2.setStyleSheet(
            "QPushButton { color: #2f9e57; }"
            "QPushButton:disabled { color: #777777; }")
        self.btn_stage2.setToolTip("Run Shape-1024 upscale")
        self.btn_stage2.clicked.connect(lambda: self.win.pipeline.run_stage2())
        cello_row.addStretch(1)
        cello_row.addWidget(self.btn_stage2)
        upscale_col.addLayout(cello_row)

        # Hi3D section header (underlined), above the experiment knobs.
        hi3d_header = underlined_label("Hi3D (optional) :")
        upscale_col.addWidget(hi3d_header)

        # Hi3D experiment knobs: sampling steps per pass and the
        # occupied-voxel budget scale. Both trade quality for speed/VRAM.
        # Row 1: steps + pass dropdown.
        hi3d_row = QHBoxLayout()
        hi3d_row.setContentsMargins(0, 0, 0, 0)
        hi3d_label = QLabel("steps")
        hi3d_row.addWidget(hi3d_label)
        self.hi3d_steps = QSpinBox()
        self.hi3d_steps.setRange(4, 50)
        self.hi3d_steps.setValue(25)
        self.hi3d_steps.setMinimumWidth(60)
        self.hi3d_steps.setToolTip("Sampling steps per Hi3D pass")
        hi3d_row.addWidget(self.hi3d_steps)
        hi3d_row.addSpacing(12)

        # Hi3D pass selection: which ladder rungs to run. Each rung refines at
        # its structure resolution and outputs its own mesh resolution
        # (64 -> 512, 128 -> 1024, 192 -> 1536); a "N pass" entry runs just
        # that rung, a "X-Y pass" entry runs rungs X through Y.
        self.hi3d_pass_combo = QComboBox()
        self.hi3d_pass_combo.setFixedHeight(28)
        self.hi3d_pass_combo.setMinimumWidth(65)
        for label, start, resolution in (
            ("1-3 pass", None, 1536),
            ("2-3 pass", 128, 1536),
            ("1 pass", 64, 512),
            ("2 pass", 128, 1024),
            ("3 pass", 192, 1536),
        ):
            self.hi3d_pass_combo.addItem(label, (start, resolution))
        self.hi3d_pass_combo.setCurrentIndex(4)   # default: 3 pass
        self.hi3d_pass_combo.setToolTip("Select Hi3D passes")
        hi3d_row.addWidget(self.hi3d_pass_combo)
        hi3d_row.addStretch(1)
        upscale_col.addLayout(hi3d_row)

        # Row 2: voxel budget + full-GPU switch + the run button.
        hi3d_row2 = QHBoxLayout()
        hi3d_row2.setContentsMargins(0, 0, 0, 0)
        hi3d_label2 = QLabel("budget")
        hi3d_row2.addWidget(hi3d_label2)
        self.hi3d_budget = QSpinBox()
        self.hi3d_budget.setRange(25, 100)
        self.hi3d_budget.setValue(100)
        self.hi3d_budget.setMinimumWidth(60)
        self.hi3d_budget.setSuffix("%")
        self.hi3d_budget.setToolTip("Hi3D voxel budget scale")
        hi3d_row2.addWidget(self.hi3d_budget)
        hi3d_row2.addSpacing(12)
        self.hi3d_fullgpu = QCheckBox("Full GPU")
        self.hi3d_fullgpu.setMinimumWidth(60)
        self.hi3d_fullgpu.setToolTip("Keep Hi3D fully on GPU")
        hi3d_row2.addWidget(self.hi3d_fullgpu)
        hi3d_row2.addStretch(1)
        upscale_col.addLayout(hi3d_row2)

        # Run button on its own row, aligned to the right edge.
        hi3d_row3 = QHBoxLayout()
        hi3d_row3.setContentsMargins(0, 0, 0, 0)
        self.btn_hi3d = QPushButton("Run Hi3D 1536")
        self.btn_hi3d.setFixedHeight(28)
        self.btn_hi3d.setMinimumWidth(80)
        self.btn_hi3d.setStyleSheet(
            "QPushButton { color: #2f9e57; }"
            "QPushButton:disabled { color: #777777; }")
        self.btn_hi3d.setToolTip("Refine mesh with Hi3D")
        self.btn_hi3d.clicked.connect(lambda: self.win.pipeline.run_hi3d())
        hi3d_row3.addStretch(1)
        hi3d_row3.addWidget(self.btn_hi3d)
        upscale_col.addLayout(hi3d_row3)

        layout.addWidget(self.grp_upscale)

        # Log ---------------------------------------------------------------
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(5000)
        layout.addWidget(self.log_box, 1)

        self.win.statusBar().showMessage("Ready.")

    # -- image thumbnails ------------------------------------------------
    def rebuild_thumbs(self):
        win = self.win
        self.thumb_list.clear()
        for path in win.app_state.image_paths:
            item = QListWidgetItem()
            item.setSizeHint(QSize(THUMB_SIZE + 4, THUMB_SIZE + 4))
            self.thumb_list.addItem(item)
            thumb = _Thumb(path, removable=True, selectable=True)
            thumb.remove.connect(self._remove_image)
            thumb.selected.connect(self._on_reference_selected)
            thumb.set_selected(path == win.app_state.ref_reference)
            self.thumb_list.setItemWidget(item, thumb)
        self._rebuild_cutouts()
        win._refresh_buttons()

    def _rebuild_cutouts(self):
        win = self.win
        self.cutout_list.clear()
        items = [win.app_state.cutout_map[p] for p in win.app_state.image_paths
                 if p in win.app_state.cutout_map]
        self.cutout_label.setText(
            f"Background removed ({len(items)}/{len(win.app_state.image_paths)}):"
            if items else "")
        self.cutout_label.setVisible(bool(items))
        self.cutout_list.setVisible(bool(items))
        for path in items:
            item = QListWidgetItem()
            item.setSizeHint(QSize(THUMB_SIZE, THUMB_SIZE))
            self.cutout_list.addItem(item)
            self.cutout_list.setItemWidget(
                item, _Thumb(path, removable=False, border="#7bd88f"))

    def _remove_image(self, path: Path):
        state = self.win.app_state
        was_reference = path == state.ref_reference
        if state.remove_image(path) and was_reference:
            self.win.log("Reference cleared (image removed).", "warn")

    def _on_thumbnail_item_clicked(self, item):
        row = self.thumb_list.row(item)
        if 0 <= row < len(self.win.app_state.image_paths):
            self._on_reference_selected(self.win.app_state.image_paths[row])

    def _on_reference_selected(self, path):
        win = self.win
        path = Path(path)
        if path not in win.app_state.image_paths or path == win.app_state.ref_reference:
            return
        win.app_state.ref_reference = path
        self._update_reference_markers()
        cutout = win.app_state.cutout_map.get(path)
        note = (" (using existing background cutout)"
                if cutout is not None and cutout.is_file() else "")
        win.log(f"Reference image: {path.name}{note}.", "ok")
        win._refresh_buttons()

    def _update_reference_markers(self):
        for i in range(self.thumb_list.count()):
            widget = self.thumb_list.itemWidget(self.thumb_list.item(i))
            if isinstance(widget, _Thumb):
                widget.set_selected(widget.path == self.win.app_state.ref_reference)

    def set_state(self, state):
        self.btn_generate.setEnabled(state.idle and state.has_images)
        self.btn_generate_texture.setVisible(state.has_texture_state)
        self.btn_generate_texture.setEnabled(state.idle and state.has_images)
        # The file-based stages lock while an unsaved edit is pending.
        upscale_on = state.idle and state.has_model and not state.pending_edit
        for widget in (self.btn_cellocut, self.btn_stage2, self.hi3d_steps,
                       self.hi3d_pass_combo, self.hi3d_budget,
                       self.hi3d_fullgpu, self.btn_hi3d):
            widget.setEnabled(upscale_on)
        self.upscale_scene.setEnabled(state.has_model)
        self.btn_stop.setEnabled(state.busy or state.refining
                                 or state.lato2 or state.repairing)
        self.btn_rmbg.setEnabled(state.idle and state.has_images)
        self.low_vram_check.setEnabled(state.idle)
        self.stop_stage1_check.setEnabled(state.idle)
        self.mode_combo.setEnabled(
            state.idle and not self.stop_stage1_check.isChecked())

    def on_seed_button(self):
        """Toggle between the last used seed and a random seed."""
        if self.seed_spin.value() < 0:
            if self.win.app_state.last_seed is None:
                return
            self.seed_spin.setValue(self.win.app_state.last_seed)
            self.win.log(f"Seed set to last used: {self.win.app_state.last_seed}.", "info")
        else:
            self.seed_spin.setValue(-1)
            self.win.log("Seed reset to random.", "info")

    def _update_seed_button(self, *_):
        if self.seed_spin.value() < 0:
            self.btn_seed.setText("Use last")
            self.btn_seed.setEnabled(self.win.app_state.last_seed is not None)
            self.btn_seed.setToolTip("Reuse the last run's seed."
                if self.win.app_state.last_seed is None
                else f"Reuse last seed ({self.win.app_state.last_seed}).")
        else:
            self.btn_seed.setText("random")
            self.btn_seed.setEnabled(True)
            self.btn_seed.setToolTip("Random new seed")

    def _on_stop_stage1_toggled(self, checked: bool):
        if checked:
            # A Shape-512-only export cannot be textured, so force the
            # geometry-only output mode while early stop is active.
            self._stop_stage1_previous_mode = self.mode_combo.currentIndex()
            self.mode_combo.setCurrentIndex(1)
            self.win.log("Stop at 1st stage enabled: Shape-512 mesh will be "
                     "decoded directly; Shape-1024 and texture are skipped.",
                     "info")
        else:
            self.mode_combo.setCurrentIndex(self._stop_stage1_previous_mode)
            self.win.log("Stop at 1st stage disabled: the full pipeline will "
                     "run.", "info")
        self.win._refresh_buttons()
