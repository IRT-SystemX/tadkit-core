import json


# =====================================================
# 1. Value Sanitization Utility
# =====================================================
def sanitize_default(default, min_val, max_val, ptype, closed="both", allow_none=False):
    """Clamp and adjust defaults based on type and bounds."""

    def cast(x):
        try:
            return None if x is None else ptype(x)
        except (ValueError, TypeError):
            return None

    d, lo, hi = map(cast, (default, min_val, max_val))
    if allow_none and d is None:
        return None
    eps = 1e-12 if ptype is float else 1

    if d is None:
        # no default, fall back to min or type zero
        if lo is not None:
            return lo + eps if closed in ("right", "neither") else lo
        return 0 if ptype is int else 0.0

    # Clamp lower bound
    if lo is not None:
        if closed in ("right", "neither") and d <= lo:
            d = lo + eps
        elif closed in ("both", "left") and d < lo:
            d = lo

    # Clamp upper bound
    if hi is not None:
        if closed in ("left", "neither") and d >= hi:
            d = hi - eps
        elif closed in ("both", "right") and d > hi:
            d = hi

    return d


# =====================================================
# 2. Widget Factory
# =====================================================
class WidgetFactory:
    """Factory for creating UI widgets or no-UI representations."""

    WIDGET_STYLE = {"description_width": "initial"}

    def __init__(self, frontend="ipywidgets"):
        self.frontend = frontend
        self._import_ui_libs()

    def _import_ui_libs(self):
        """Lazily import UI libraries."""
        if self.frontend == "st":
            import streamlit as st

            self.st = st
        elif self.frontend == "ipywidgets":
            import ipywidgets as widgets

            self.widgets = widgets

    # -------------------------------------------------
    # Generic dispatcher for UI elements
    # -------------------------------------------------
    def _dispatch(self, mapping, **kwargs):
        func = mapping.get(self.frontend)
        if func is None:
            raise ValueError(f"Unsupported frontend: {self.frontend}")
        return func(**kwargs)

    # -------------------------------------------------
    # Numeric input
    # -------------------------------------------------
    def make_numeric(
        self,
        label,
        default,
        min_val,
        max_val,
        ptype,
        description,
        closed="both",
        allow_none=False,
    ):
        """Create a numeric input widget."""
        # Sanitize default
        if not (allow_none and default is None):
            default = sanitize_default(
                default, min_val, max_val, ptype, closed, allow_none
            )

        # Streamlit
        def _st_numeric(**kw):
            st = self.st
            step = 1 if ptype is int else 0.01
            fmt = "%d" if ptype is int else "%.6f"

            def cast(x):
                return None if x is None else ptype(x)

            args = {
                "label": label,
                "value": cast(default),
                "step": cast(step),
                "format": fmt,
                "help": description,
            }
            if min_val is not None:
                args["min_value"] = cast(min_val)
            if max_val is not None:
                args["max_value"] = cast(max_val)
            return st.number_input(**args)

        # ipywidgets
        def _ipy_numeric(**kw):
            w = self.widgets
            cls = w.BoundedIntText if ptype is int else w.BoundedFloatText
            return cls(
                value=default,
                min=min_val if min_val is not None else -1e6,
                max=max_val if max_val is not None else 1e6,
                description=label,
                style=self.WIDGET_STYLE,
                tooltip=description,
            )

        # No UI
        def _noui_numeric(**kw):
            return {
                "type": "number",
                "label": label,
                "default": default,
                "min": min_val,
                "max": max_val,
                "description": description,
                "nullable": allow_none,
            }

        return self._dispatch(
            {
                "st": _st_numeric,
                "ipywidgets": _ipy_numeric,
                "no_ui": _noui_numeric,
            }
        )

    # -------------------------------------------------
    # Dropdown
    # -------------------------------------------------
    def make_dropdown(self, label, options, default, description):
        """Create a dropdown input."""
        options = options or ["(none)"]
        default_val = default if default in options else options[0]

        return self._dispatch(
            {
                "st": lambda **kw: self.st.selectbox(
                    label, options, index=options.index(default_val), help=description
                ),
                "ipywidgets": lambda **kw: self.widgets.Dropdown(
                    options=options,
                    value=default_val,
                    description=label,
                    style=self.WIDGET_STYLE,
                    tooltip=description,
                ),
                "no_ui": lambda **kw: {
                    "type": "dropdown",
                    "label": label,
                    "options": options,
                    "default": default_val,
                    "description": description,
                },
            }
        )

    # -------------------------------------------------
    # Text input
    # -------------------------------------------------
    def make_text(self, label, default, description):
        value = str(default or "")
        return self._dispatch(
            {
                "st": lambda **kw: self.st.text_input(
                    label, value=value, help=description
                ),
                "ipywidgets": lambda **kw: self.widgets.Text(
                    value=value,
                    description=label,
                    style=self.WIDGET_STYLE,
                    tooltip=description,
                ),
                "no_ui": lambda **kw: {
                    "type": "text",
                    "label": label,
                    "default": default,
                    "description": description,
                },
            }
        )

    # -------------------------------------------------
    # Dictionary / JSON input
    # -------------------------------------------------
    def make_dict(self, label, default, description):
        json_str = json.dumps(default or {}, indent=2)
        return self._dispatch(
            {
                "st": lambda **kw: self.st.text_area(
                    label, value=json_str, help=description, height=100
                ),
                "ipywidgets": lambda **kw: self.widgets.Textarea(
                    value=json_str,
                    description=label,
                    layout=self.widgets.Layout(width="100%", height="80px"),
                    tooltip=description,
                ),
                "no_ui": lambda **kw: {
                    "type": "dict",
                    "label": label,
                    "default": default,
                    "description": description,
                },
            }
        )
