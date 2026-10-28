"""Outliner column: one button per model file, nested by derived names.

Model files are every ``*.glb`` in the folder plus any ``*.obj`` that has no
same-stem GLB (see ``paths._folder_models``).
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMenu, QPushButton, QScrollArea,
    QToolButton, QVBoxLayout, QWidget,
)

from paths import _folder_models, _paths_equal
from utils.layout import underlined_label
from widgets import _ElidedButton


class OutlinerPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        panel = self
        panel.setObjectName("objectsPanel")
        panel.setMinimumWidth(96)
        panel.setMaximumWidth(420)
        panel.setStyleSheet(
            "QWidget#objectsPanel { background: #1e2022; }")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 0, 6, 8)   # no top padding
        layout.setSpacing(6)

        # Draw toggle row, above the "Outliner:" header: a small square
        # checkable button pinned to the right edge.  OFF by default; the
        # smooth brush's Shift-hold activation only works while it is on
        # (see viewer_brush.BrushController.set_draw_enabled).
        draw_row = QWidget()
        draw_layout = QHBoxLayout(draw_row)
        draw_layout.setContentsMargins(0, 0, 0, 0)
        draw_layout.addStretch(1)
        self.draw_button = QPushButton("Draw")
        self.draw_button.setCheckable(True)
        self.draw_button.setChecked(False)
        self.draw_button.setFixedSize(28, 28)   # square: width == height
        self.draw_button.setCursor(Qt.PointingHandCursor)
        self.draw_button.setToolTip(
            "Draw mode: circle cursor + smooth brush (Q).\n"
            "The ring follows the mouse everywhere; hold Shift over the "
            "mesh and left-drag to sculpt. W / S resize the ring.")
        self.draw_button.setStyleSheet(
            "QPushButton {"
            " background: #2a2d31; color: #9aa0a6;"
            " border: 1px solid #3a3e44; border-radius: 4px;"
            " font-size: 9px; padding: 0; }"
            "QPushButton:hover { background: #33373c; }"
            "QPushButton:checked { background: #3a3e44; color: #e6e8eb;"
            " border-color: #565b62; }")
        self.draw_button.toggled.connect(self._on_draw_toggled)
        draw_layout.addWidget(self.draw_button, 0, Qt.AlignTop)
        layout.addWidget(draw_row, 0, Qt.AlignTop)

        header = underlined_label("Outliner:")
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollBar:vertical { background: #1d1f22; width: 8px; }"
            "QScrollBar::handle:vertical { background: #4a4e55;"
            " min-height: 20px; border-radius: 3px; }")
        self.objects_box = QWidget()
        self.objects_box.setObjectName("objectsBox")
        self.objects_box.setStyleSheet(
            "QWidget#objectsBox { background: transparent; }")
        self.objects_layout = QVBoxLayout(self.objects_box)
        self.objects_layout.setContentsMargins(0, 0, 0, 0)
        self.objects_layout.setSpacing(4)
        scroll.setWidget(self.objects_box)
        layout.addWidget(scroll, 1)

        self._show_buttons: list[QPushButton] = []

    def _on_draw_toggled(self, checked: bool):
        """Forward the Draw toggle to the brush controller's gate."""
        self.win.viewer.brush.set_draw_enabled(checked)

    def sync_draw_button(self, checked: bool):
        """Mirror a viewer-driven Draw change (model swap) on the button.

        Signals are blocked so the uncheck does not loop back through
        ``_on_draw_toggled``; the :checked stylesheet restyles it anyway.
        """
        self.draw_button.blockSignals(True)
        self.draw_button.setChecked(checked)
        self.draw_button.blockSignals(False)

    def _show_object_menu(self, path, global_pos):
        menu = QMenu(self)
        act_rename = menu.addAction("Rename...")
        act_show_obj = menu.addAction("Show obj")
        obj_path = self.win.models.obj_path(path)
        act_show_obj.setEnabled(obj_path is not None)
        act_show_obj.setToolTip(
            f"Select {obj_path.name} in Explorer." if obj_path is not None
            else "No matching .obj file beside this model.")
        act_duplicate = menu.addAction("Duplicate")
        act_delete = menu.addAction("Delete")
        menu.setToolTipsVisible(True)
        chosen = menu.exec(global_pos)
        if chosen is act_rename:
            self.win.models.rename(path)
        elif chosen is act_show_obj:
            self.win.models.show_obj(path)
        elif chosen is act_duplicate:
            self.win.models.duplicate(path)
        elif chosen is act_delete:
            self.win.models.delete(path)

    def refresh(self):
        folder = self.win.models.folder
        # GLBs plus OBJs that have no GLB twin, so import-only assets show up.
        # Natural order: prediction_512 before prediction_1024 before
        # prediction_hi3d_1536 etc. sorted() already does the right thing for
        # our naming scheme.
        entries = (_folder_models(folder)
                   if folder and folder.is_dir() else [])
        current = self.win.viewer.current_path
        current_key = current.resolve() if current is not None else None

        # Clear everything the previous refresh put in the column.  Reparent
        # immediately so the old buttons/hint vanish before the deferred
        # deleteLater() runs (otherwise they can paint over the new list).
        while self.objects_layout.count():
            item = self.objects_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        self._show_buttons.clear()

        if not entries:
            hint = QLabel(
                "Open a folder to list models."
                if folder is None
                else "No *.glb or *.obj files in this folder.")
            hint.setWordWrap(True)
            hint.setStyleSheet(
                "color: #888c92; font-size: 10px; padding: 2px;")
            self.objects_layout.addWidget(hint)
            self.objects_layout.addStretch(1)
            return

        # Group derived model files under their base object.  A file whose
        # stem is "<other stem>_<suffix>" (e.g. obj_512_cellocut) nests under
        # the file named obj_512, but only if obj_512.glb (or the unpaired
        # obj_512.obj) actually exists.
        # Nesting follows the names all the way down: obj_512_cellocut_hi
        # nests under obj_512_cellocut as "└─ hi", each level showing only
        # the part of the name that was matched away.
        stems = {entry.stem for entry in entries}
        lowered = {s: s.lower() for s in stems}
        parent_of = {}
        suffix_of = {}
        for entry in entries:
            s = lowered[entry.stem]
            best = None
            for other, o in lowered.items():
                if o != s and s.startswith(o + "_"):
                    if best is None or len(other) > len(best):
                        best = other        # longest matching base wins
            parent_of[entry.stem] = best
            suffix_of[entry.stem] = (
                entry.stem[len(best) + 1:] if best else entry.stem)

        roots = []
        children = {}       # base stem -> [entry, ...] in sorted order
        for entry in entries:
            p = parent_of[entry.stem]
            if p is None:
                roots.append(entry)
            else:
                children.setdefault(p, []).append(entry)

        def _make_btn(entry, level):
            text = (entry.stem if level == 0
                    else f"└─ {suffix_of[entry.stem]}")
            shown = current_key == entry.resolve()
            btn = _ElidedButton(text)
            btn.setFixedHeight(24)
            btn.setCursor(Qt.PointingHandCursor)
            if level:
                # Indented, slightly smaller/dimmer; active stays green.
                pad = 10 + 10 * (level - 1)
                btn.setStyleSheet(
                    "QPushButton { text-align: left; padding: 2px 6px 2px"
                    f" {pad}px; font-size: 10px; color: #9aa0a6; }}"
                    + ("QPushButton { color: #2f9e57; }" if shown else ""))
            else:
                btn.setStyleSheet(
                    "QPushButton { text-align: left; padding: 2px 6px;"
                    " font-size: 11px; }"
                    + ("QPushButton { color: #2f9e57; }" if shown else ""))
            tip = (f"{entry.name}\nCurrently displayed." if shown
                   else f"Show {entry.name} in the 3D view (camera kept).")
            btn.setToolTip(tip + "\nMiddle-click to ghost-compare; "
                           "right-click for more.")
            btn.clicked.connect(
                lambda _=False, p=entry: self.win.models.show(p))
            btn.middleClicked.connect(
                lambda p=entry: self.win.models.ghost(p))
            ghost = self.win.viewer.ghost_path
            btn.set_dot(ghost is not None and _paths_equal(ghost, entry))
            # Keep the active entry enabled so it still accepts right-clicks;
            # on_show_model treats clicking it as a no-op.
            btn.setContextMenuPolicy(Qt.CustomContextMenu)
            btn.customContextMenuRequested.connect(
                lambda pos, p=entry, b=btn:
                    self._show_object_menu(p, b.mapToGlobal(pos)))
            return btn

        def _build_branch(entry, level):
            """(widget, buttons) for entry followed by everything under it."""
            btn = _make_btn(entry, level)
            btns = [btn]
            child_widgets = []
            for child in children.get(entry.stem, ()):
                cw, cb = _build_branch(child, level + 1)
                child_widgets.append(cw)
                btns.extend(cb)
            if not child_widgets:
                return btn, btns

            # Every row with children gets an accordion arrow, at any
            # nesting level, so any branch can be collapsed independently.
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(2)
            rl.addWidget(btn, 1)
            arrow = QToolButton()
            arrow.setFixedSize(16, 24)
            arrow.setCursor(Qt.PointingHandCursor)
            arrow.setToolTip("Collapse / expand this object's branch.")
            arrow.setStyleSheet(
                "QToolButton { border: none; background: transparent;"
                " color: #9aa0a6; font-size: 9px; }"
                "QToolButton:hover { color: #d0d3d8; }")
            rl.addWidget(arrow, 0)

            branch = QWidget()
            bl = QVBoxLayout(branch)
            bl.setContentsMargins(0, 0, 0, 0)
            bl.setSpacing(4)
            for cw in child_widgets:
                bl.addWidget(cw)

            container = QWidget()
            cl = QVBoxLayout(container)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(4)
            cl.addWidget(row)
            cl.addWidget(branch)

            state = {"open": True}

            def _toggle():
                state["open"] = not state["open"]
                # Hiding the branch container leaves each nested branch's
                # own open/closed state untouched.
                branch.setVisible(state["open"])
                arrow.setText("▼" if state["open"] else "▶")

            arrow.setText("▼")
            arrow.clicked.connect(_toggle)
            return container, btns

        for root in roots:
            widget, btns = _build_branch(root, 0)
            self.objects_layout.addWidget(widget)
            self._show_buttons.extend(btns)
        self.objects_layout.addStretch(1)
