"""UniForge3D - PySide6 + VTK desktop UI for image-to-3D generation.

Native Windows app (no browser): PySide6 controls on the left, an embedded
VTK 3D viewer (GLB/OBJ/STL/PLY with PBR materials) on the right.

Feature parity with the Flet UI:
  - imported-image thumbnails with per-image remove buttons
  - pre-generation background removal (RMBG-2.0 in a light worker process),
    with the resulting RGBA cutouts shown as thumbnails of their own; Generate
    runs this automatically for any image that has no cutout yet
  - textured / geometry-only mode, grid, texture size, steps, seed, the
    Low VRAM switch (on by default; off runs fp32 weights + bf16 compute),
    and an optional early stop after Shape-512 (skips the 1024 upscale);
    Run 2nd stage can later resume the 1024 upscale from that saved mesh,
    or from any currently displayed mesh.  The upscale normalizes its source
    into UniForge's canonical cube; when that changed an imported mesh's
    size/location, the "Uniforge 2nd stage Rescale" setting (Settings dialog,
    on by default) automatically applies the inverse to the 1024 result and
    overwrites the saved stage-2 mesh with it (off leaves the result as-is).
  - Open folder fills the slim Outliner column (attached to the left
    controls) with one button per *.glb in the folder (plus any *.obj that
    has no same-stem GLB), so any stage of a result can be viewed (camera
    and material preserved) and used directly:
    Run CelloCut, Run 2nd stage and Run Hi3D 1536 all operate on the
    displayed model.  Right-clicking an entry offers Rename..., Show obj,
    Duplicate and Delete; Show obj selects the matching .obj sibling in
    Explorer, and a renamed/deleted entry takes its .shape512.pt/
    .shape_state.pt sidecars with it (a Duplicate copies only the GLB).
    The controls|Outliner, Outliner|view and view|edit boundaries are all
    draggable splitters.
  - Run Hi3D 1536 refines the displayed mesh with Hi3D v1.0
    (image-conditioned, geometry only); a dropdown picks the passes:
    1-3, 2-3, 1, 2 or 3. Each rung outputs its own resolution
    (64 -> 512, 128 -> 1024, 192 -> 1536), saved as
    <model>_hi3d_<out>.glb next to the source. It runs in one-shot
    workers/hi3d_worker.py so its transformers/VAE unload immediately afterwards;
    the decode phase reports per-stage progress in the log.
  - Run LATO.2 retopologizes the displayed mesh with LATO.2 (V-Flow +
    T-Flow), using a controllable target vertex count. It conditions on a
    white-model render of the source mesh, saves <model>_lato_<verts>.glb
    plus its .shape512.pt checkpoint for Run 2nd stage, and runs in
    one-shot workers/lato2_worker.py so its ~5 GB of weights unload
    immediately afterwards.
  - Run CelloCut (watertight) repairs the displayed mesh with CelloCut
    (constructive 512^3 UDF + graph-cut remeshing; no model download),
    saving <model>_cellocut.glb plus a regenerated <model>_cellocut
    .shape512.pt state next to it - the source mesh is untouched, and
    Run 2nd stage / Hi3D can continue from the repaired geometry.
  - The scrollable edit side panel (right of the 3D view) stacks three
    geometry operations in memory:
      * Reproject: copies the ghosted donor mesh's detail (middle-click an
        entry in the Outliner column; Ctrl+1 hides/shows the overlay) onto
        the displayed mesh with four modes (exact ProjectAll snap,
        normal-only, high-pass detail, higher-subdiv snap).
      * Polish: umbrella smoothing plus 5..25 tangential relaxation passes.
      * Decimate: quadric-error reduction to a target triangle count
        (tools/decimate; open3d or fast_simplification backend).
      * Remesh: native Python quad remesher (tools/remesh) rebuilds the
        displayed geometry as a quad-dominant grid.
    Running any operation only updates the view; the pinned Save button
    below the panel writes the combined result as <model>_edit.glb.

Generation runs in a separate worker process (workers/worker.py) so
Stop can hard-kill a run and this process stays lightweight (no torch import).
Background removal uses workers/rmbg_worker.py, which only loads RMBG-2.0 (~never the
26 GB pipeline).

Run with:  .venv/Scripts/python.exe PS6_UI/main.py   (or start_ps6.bat)
"""

import sys
from pathlib import Path

# The hot-reloadable tools package (tools/_loader.py, tools/mesh_edit/...)
# lives in the project root, one level above this directory.  Put the root
# on sys.path before anything imports `tools`, or the import machinery
# can't find tools._loader and every panel tool fails to load.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QSplitter, QVBoxLayout, QWidget,
)

import viewer as viewer_mod

from paths import IMAGE_SUFFIXES, OUTPUT_DIR, ROOT, _paths_equal
import hotkeys
from app.state import ActionState, AppState
from app.editing import EditSession
from app.jobs import JobCoordinator
from app.models import ModelLibrary
from app.pipeline import Pipeline
from panels import LeftPanel, OutlinerPanel, RightPanel
from utils.theme import DARK_PALETTE, SPLITTER_STYLE
from dialogs import SettingsDialog, app_settings as _app_settings

# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UniForge3D - image to 3D")
        self.resize(1500, 900)
        self.setPalette(DARK_PALETTE)

        self.app_state = AppState(self)
        self._build_jobs()

        self._build_menu()
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        split_style = SPLITTER_STYLE

        # Every vertical panel boundary is draggable:
        #   controls | Outliner | 3D view | edit panel
        # The left controls start slightly narrower than before and the
        # Outliner column is free to grow/shrink instead of being fixed.
        self.left_split = QSplitter(Qt.Horizontal)
        self.left_split.setChildrenCollapsible(False)
        self.left_split.setHandleWidth(4)
        self.left_split.setStyleSheet(split_style)
        self.left_panel = LeftPanel(self)
        self.outliner = OutlinerPanel(self)
        self.left_split.addWidget(self.left_panel)
        self.left_split.addWidget(self.outliner)
        self.left_split.setStretchFactor(0, 1)   # extra room -> controls/log
        self.left_split.setStretchFactor(1, 0)   # Outliner keeps its width
        self.left_split.setSizes([300, 152])

        self.root_split = QSplitter(Qt.Horizontal)
        self.root_split.setChildrenCollapsible(False)
        self.root_split.setHandleWidth(4)
        self.root_split.setStyleSheet(split_style)
        self.root_split.addWidget(self.left_split)
        self.root_split.addWidget(self._build_viewer())
        self.root_split.setStretchFactor(0, 0)   # window growth -> 3D view
        self.root_split.setStretchFactor(1, 1)
        self.root_split.setSizes([456, 1024])
        root.addWidget(self.root_split)
        self._build_domains()
        self._refresh_buttons()
        self.outliner.refresh()

    # -- background jobs ---------------------------------------------------
    def _build_jobs(self):
        self.jobs = JobCoordinator(self)
        self.jobs.log.connect(self.log)
        self.jobs.changed.connect(self._refresh_buttons)
        self.jobs.edit_failed.connect(self._on_edit_failed)

    # -- domain services ---------------------------------------------------
    def _build_domains(self):
        self.edits = EditSession(self.viewer, self.jobs, self)
        # Model swaps turn Draw mode off inside the viewer; keep the
        # Outliner panel's Draw button in sync (signals blocked there).
        self.viewer.drawModeChanged.connect(self.outliner.sync_draw_button)
        self.models = ModelLibrary(self.viewer, self.jobs, self.edits, self)
        self.pipeline = Pipeline(self, self.viewer, self.jobs, self.models,
                                 self.edits, self.app_state, self.left_panel,
                                 self)
        self.viewer.smoothEdited.connect(self.edits.capture_smooth_brush)
        self.jobs.edit_done.connect(self.edits.accept_result)
        for signal in (self.edits.log, self.models.log, self.pipeline.log):
            signal.connect(self.log)
        self.edits.failed.connect(self._on_job_failed)
        self.edits.changed.connect(self._on_edit_changed)
        self.edits.saved.connect(self._on_edit_saved)
        self.edits.folder_changed.connect(self.models.set_folder)
        self.models.changed.connect(self._on_models_changed)
        self.pipeline.changed.connect(self._refresh_buttons)
        self.app_state.changed.connect(self._on_images_changed)
        self.app_state.seed_changed.connect(
            self.left_panel._update_seed_button)
        self.jobs.remesh_progress.connect(
            self.right_panel.on_remesh_progress)
        self.jobs.cutout_ready.connect(self.pipeline.on_cutout_ready)
        self.jobs.rmbg_finished.connect(self.pipeline.on_rmbg_finished)
        self.jobs.generation_done.connect(self.pipeline.on_generation_done)
        self.jobs.generation_failed.connect(
            self.pipeline.on_generation_failed)
        self.jobs.hi3d_done.connect(self.pipeline.on_hi3d_done)
        self.jobs.hi3d_failed.connect(self.pipeline.on_hi3d_failed)
        self.jobs.lato2_done.connect(self.pipeline.on_lato2_done)
        self.jobs.lato2_failed.connect(self.pipeline.on_lato2_failed)
        self.jobs.cellocut_done.connect(self.pipeline.on_cellocut_done)
        self.jobs.cellocut_failed.connect(self.pipeline.on_cellocut_failed)

    def _on_images_changed(self):
        self.left_panel.rebuild_thumbs()

    def _on_edit_changed(self):
        self.right_panel.refresh_edit_state()
        self._refresh_buttons()

    def _on_edit_saved(self, output, shown):
        if shown:
            self.models.adopt(output)
        else:
            self.models.track_output(output)

    def _on_models_changed(self):
        self.outliner.refresh()
        self._refresh_buttons()

    # -- UI construction ------------------------------------------------
    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        act_open = QAction("Open &folder...", self)
        act_open.setShortcut(hotkeys.seq("open_folder"))
        act_open.triggered.connect(self.on_open_folder)
        m.addAction(act_open)
        m.addSeparator()
        act_quit = QAction("E&xit", self)
        act_quit.setShortcut(hotkeys.seq("exit"))
        act_quit.triggered.connect(self.close)
        m.addAction(act_quit)

        view = self.menuBar().addMenu("&View")
        act_reset = QAction("Reset &view", self)
        act_reset.setToolTip("Refit camera to the model")
        act_reset.triggered.connect(self.viewer_reset)
        view.addAction(act_reset)
        view.addSeparator()
        self.act_backface = QAction("&Backface culling", self)
        self.act_backface.setCheckable(True)
        self.act_backface.setChecked(True)   # default ON (before connecting)
        self.act_backface.setToolTip("Cull back-facing polygons")
        self.act_backface.toggled.connect(
            lambda checked: self.viewer.set_backface_culling(checked))
        view.addAction(self.act_backface)
        view.addSeparator()
        # Ctrl+1: hide/show the ghost overlay mesh (middle-clicked in the
        # Outliner column) without dropping the ghost selection - reproject
        # keeps using it as the donor while it is hidden.
        self.act_ghost_visible = QAction("&Ghost mesh visible", self)
        self.act_ghost_visible.setCheckable(True)
        self.act_ghost_visible.setChecked(True)   # default ON (before connect)
        self.act_ghost_visible.setShortcut(hotkeys.seq("ghost_visible"))
        self.act_ghost_visible.setEnabled(False)  # no ghost picked yet
        self.act_ghost_visible.setToolTip("Show or hide the ghost")
        self.act_ghost_visible.toggled.connect(
            lambda visible: self.viewer.set_ghost_visible(visible))
        view.addAction(self.act_ghost_visible)

        settings_menu = self.menuBar().addMenu("&Settings")
        act_settings = QAction("&General...", self)
        act_settings.setShortcut(hotkeys.seq("settings"))
        act_settings.triggered.connect(self.on_open_settings)
        settings_menu.addAction(act_settings)

    def on_open_settings(self):
        """Show the Settings dialog and persist the chosen options."""
        dialog = SettingsDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        enabled = dialog.stage2_rescale.isChecked()
        _app_settings().setValue("settings/stage2_rescale", enabled)
        self.log(
            "Settings saved: auto-rescale for 2nd stage and Hi3D is "
            f"{'ON' if enabled else 'OFF'}.", "info")

    def _build_viewer(self) -> QWidget:
        panel = QWidget()
        row = QHBoxLayout(panel)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        # 3D view with its toolbar; the edit side panel (Reproject, Polish,
        # Remesh) sits to the right so its controls stay next to the model.
        view = QWidget()
        v = QVBoxLayout(view)
        v.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        b_open = QPushButton("Open folder")
        b_open.setToolTip("Open a model folder")
        b_open.clicked.connect(self.on_open_folder)
        self.btn_folder = QPushButton("Open output folder")
        self.btn_folder.clicked.connect(self.on_open_output_folder)
        self.btn_folder.setVisible(False)
        bar.addSpacing(12)               # left offset before "Open folder"
        bar.addWidget(b_open)
        bar.addWidget(self.btn_folder)
        bar.addStretch(1)
        # The per-model buttons live in the Outliner column attached to the
        # left controls. Reset view and Backface culling are in the View menu.
        self.row_status = QLabel("● Idle")
        self.row_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.row_status.setStyleSheet(
            "color: #7bd88f; font-size: 11px;")
        self.row_status.setToolTip("Current background/GPU activity")
        bar.addWidget(self.row_status)
        bar.addSpacing(10)               # gap between status and projection toggle
        self.btn_projection = QPushButton("Perspective")
        self.btn_projection.setFixedWidth(96)
        self.btn_projection.setToolTip("Toggle perspective or orthographic")
        self.btn_projection.clicked.connect(self.on_toggle_projection)
        bar.addWidget(self.btn_projection)
        bar.addSpacing(10)               # gap between projection toggle and edge
        bar.addSpacing(12)               # breathing room from the window edge
        v.addLayout(bar)

        self.viewer = viewer_mod.ModelViewer(view)
        v.addWidget(self.viewer, 1)
        view.setMinimumWidth(360)
        # Smooth-brush strokes reach EditSession via the connection made in
        # _build_domains; its pending-stack changes drive the Save footer.
        self.viewer.notice.connect(self.log)

        # The edit panel is drag-resizable rather than a fixed column, so the
        # user can trade width between it and the 3D canvas (or narrow the
        # canvas when the panel needs more room for hint text).
        self.edit_split = QSplitter(Qt.Horizontal)
        self.edit_split.setChildrenCollapsible(False)
        self.edit_split.setHandleWidth(4)
        self.edit_split.setStyleSheet(SPLITTER_STYLE)
        self.edit_split.addWidget(view)
        self.right_panel = RightPanel(self)
        self.edit_split.addWidget(self.right_panel)
        self.edit_split.setStretchFactor(0, 1)   # extra width -> 3D view
        self.edit_split.setStretchFactor(1, 0)   # panel keeps its dragged size
        self.edit_split.setSizes([810, 320])
        row.addWidget(self.edit_split, 1)
        return panel

    def _action_state(self) -> ActionState:
        loaded = self.viewer.current_path
        donor = self.viewer.ghost_path
        has_model = bool(loaded and loaded.is_file())
        return ActionState(
            busy=self.jobs.worker.generating,
            cutting=self.jobs.rmbg.running,
            refining=self.jobs.hi3d.refining,
            lato2=self.jobs.lato2.running,
            repairing=self.jobs.cellocut.running,
            reprojecting=self.jobs.reproject.running,
            polishing=self.jobs.polish.running,
            remeshing=self.jobs.remesh.running,
            decimating=self.jobs.decimate.running,
            subdividing=self.jobs.subdivide.running,
            closing_holes=self.jobs.close_holes.running,
            deleting_hidden=self.jobs.delete_faces.running,
            mirroring=self.jobs.mirror_merge.running,
            has_images=bool(self.app_state.image_paths),
            has_model=has_model,
            has_texture_state=bool(self.models.texture_state
                                   and self.models.texture_state.is_file()),
            pending_edit=self.edits.has_pending,
            donor_ok=bool(donor is not None and donor.is_file()
                          and (not has_model
                               or not _paths_equal(donor, loaded))),
            has_ghost=donor is not None,
            ghost_visible=self.viewer.ghost_visible,
            pending_kind=self.jobs.pending_kind,
        )

    def _refresh_buttons(self):
        state = self._action_state()
        self.left_panel.set_state(state)
        self.right_panel.set_state(state)
        self.act_ghost_visible.setEnabled(state.has_ghost)
        self.act_ghost_visible.blockSignals(True)
        self.act_ghost_visible.setChecked(state.ghost_visible)
        self.act_ghost_visible.blockSignals(False)
        color = "#f0c674" if state.activity != "Idle" else "#7bd88f"
        self.row_status.setText(f"● {state.activity}")
        self.row_status.setStyleSheet(f"color: {color}; font-size: 11px;")
        self.statusBar().showMessage(
            "Ready." if state.activity == "Idle" else state.activity)
        self.btn_folder.setVisible(
            state.idle and (state.has_model
                            or self.models.last_output is not None))

    # -- Handlers ---------------------------------------------------------
    def viewer_reset(self):
        if hasattr(self, "viewer"):
            self.viewer.reset_camera()

    def on_add_images(self):
        exts = " ".join("*" + s for s in sorted(IMAGE_SUFFIXES))
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add input image(s)", str(ROOT / "assets"),
            f"Images ({exts})")
        self.app_state.add_images(files)

    def on_clear_images(self):
        if self.jobs.cancel_after_rmbg():
            self.log("Pending generation cancelled (images cleared).", "warn")
        self.app_state.clear()

    def on_open_folder(self):
        """Pick a folder; its GLBs fill the Outliner column and one loads."""
        start = str(self.models.folder or
                    (OUTPUT_DIR if OUTPUT_DIR.is_dir() else ROOT))
        chosen = QFileDialog.getExistingDirectory(
            self, "Open model folder", start)
        if not chosen:
            return
        self.pipeline.open_folder(Path(chosen))

    def on_open_output_folder(self):
        """Open the folder holding the last generated model (glb/obj/...)."""
        self.models.open_output_folder()

    def on_toggle_projection(self):
        """Swap the 3D camera between perspective and orthographic."""
        mode = self.viewer.toggle_projection()
        self.btn_projection.setText(mode)

    # -- job events --------------------------------------------------------

    def _on_job_failed(self, title: str, msg: str):
        self.log(msg, "error")
        QMessageBox.warning(self, title, msg)
        self._refresh_buttons()

    def _on_edit_failed(self, title: str, msg: str):
        if msg.endswith("cancelled."):
            self.log(msg, "warn")
            self._refresh_buttons()
        else:
            self._on_job_failed(f"{title} failed", msg)

    def log(self, msg: str, level: str = "info"):
        self.left_panel.log_box.appendPlainText(msg)

    def closeEvent(self, event):
        self.jobs.shutdown()
        super().closeEvent(event)


def main() -> int:
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("UniForge3D")
    win = MainWindow()
    win.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
