"""Screen-space lasso selection and polygroup state (tools package).

Moved out of the UI folder so the viewer and any hot-reloadable tool can
share one implementation:

* ``lasso_select`` - transient screen-space face selection: the rubber-band
  overlay, projection-based hit testing and the selection highlight actors.
* ``polygroups``   - persistent per-face group ids (25 colour slots), the
  visibility filter with its clone actors, and snapshot/restore used by the
  shared Ctrl+Z history (tools/history.py).

Both are plain numpy + VTK modules with no Qt dependency, so worker tools
may import them too.
"""
