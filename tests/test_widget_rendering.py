import sys

import types
import json
import pytest

from tadkit.utils.ui import sanitize_default, WidgetFactory


# =====================================================
# 1. Tests for sanitize_default
# =====================================================
@pytest.mark.parametrize(
    "default,min_val,max_val,ptype,closed,expected",
    [
        (5, 0, 10, int, "both", 5),  # within range
        (-5, 0, 10, int, "both", 0),  # below min -> clamp
        (15, 0, 10, int, "both", 10),  # above max -> clamp
        (0, 0, 10, int, "right", 1),  # closed=right shifts up
        (10, 0, 10, int, "left", 9),  # closed=left shifts down
        (None, 0, 10, int, "both", 0),  # None -> default to 0
        (0.5, 0.0, 1.0, float, "both", 0.5),  # float ok
        (-0.5, 0.0, 1.0, float, "both", 0.0),  # clamp float min
        (1.5, 0.0, 1.0, float, "both", 1.0),  # clamp float max
    ],
)
def test_sanitize_default(default, min_val, max_val, ptype, closed, expected):
    result = sanitize_default(default, min_val, max_val, ptype, closed)
    assert (
        abs(result - expected) < 1e-10
        if isinstance(result, float)
        else result == expected
    )


def test_sanitize_default_allow_none():
    assert sanitize_default(None, 0, 10, int, allow_none=True) is None


# =====================================================
# 2. Tests for WidgetFactory (no_ui)
# =====================================================
@pytest.fixture
def wf_no_ui():
    return WidgetFactory(frontend="no_ui")


def test_make_numeric_no_ui(wf_no_ui):
    widget = wf_no_ui.make_numeric("Age", 25, 0, 100, int, "Age in years")
    assert widget["type"] == "number"
    assert widget["default"] == 25
    assert widget["min"] == 0
    assert widget["max"] == 100


def test_make_dropdown_no_ui(wf_no_ui):
    widget = wf_no_ui.make_dropdown("Color", ["red", "blue"], "blue", "Pick a color")
    assert widget["type"] == "dropdown"
    assert "blue" in widget["options"]


def test_make_text_no_ui(wf_no_ui):
    widget = wf_no_ui.make_text("Name", "Alice", "User name")
    assert widget["type"] == "text"
    assert widget["default"] == "Alice"


def test_make_dict_no_ui(wf_no_ui):
    d = {"a": 1}
    widget = wf_no_ui.make_dict("Config", d, "Settings dict")
    assert widget["type"] == "dict"
    assert widget["default"] == d


# =====================================================
# 3. Tests for WidgetFactory (ipywidgets mocked)
# =====================================================
@pytest.fixture
def wf_ipywidgets(monkeypatch):
    # Mock ipywidgets
    class DummyWidget:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyLayout:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    mock_widgets = types.SimpleNamespace(
        BoundedIntText=DummyWidget,
        BoundedFloatText=DummyWidget,
        Dropdown=DummyWidget,
        Text=DummyWidget,
        Textarea=DummyWidget,
        Layout=DummyLayout,
    )
    monkeypatch.setitem(sys.modules, "ipywidgets", mock_widgets)
    return WidgetFactory(frontend="ipywidgets")


def test_make_numeric_ipywidgets(wf_ipywidgets):
    widget = wf_ipywidgets.make_numeric("Num", 5, 0, 10, int, "desc")
    assert hasattr(widget, "kwargs")
    assert widget.kwargs["value"] == 5


def test_make_dropdown_ipywidgets(wf_ipywidgets):
    widget = wf_ipywidgets.make_dropdown("Choice", ["a", "b"], "a", "desc")
    assert "value" in widget.kwargs
    assert widget.kwargs["value"] == "a"


# =====================================================
# 4. Tests for WidgetFactory (Streamlit mocked)
# =====================================================
@pytest.fixture
def wf_streamlit(monkeypatch):
    mock_st = types.SimpleNamespace(
        number_input=lambda **kw: kw,
        selectbox=lambda *a, **kw: kw,
        text_input=lambda *a, **kw: kw,
        text_area=lambda *a, **kw: kw,
    )
    monkeypatch.setitem(sys.modules, "streamlit", mock_st)
    return WidgetFactory(frontend="st")


def test_make_numeric_streamlit(wf_streamlit):
    widget = wf_streamlit.make_numeric("Num", 5, 0, 10, int, "desc")
    assert "value" in widget
    assert widget["value"] == 5


def test_make_text_streamlit(wf_streamlit):
    widget = wf_streamlit.make_text("Name", "Bob", "desc")
    assert widget["value"] == "Bob"


def test_make_dict_streamlit(wf_streamlit):
    widget = wf_streamlit.make_dict("Config", {"x": 1}, "desc")
    assert "value" in widget
    json.loads(widget["value"])  # should be valid JSON
