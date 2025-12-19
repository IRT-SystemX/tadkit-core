import pytest
from numbers import Real, Integral
from sklearn.utils._param_validation import Interval, StrOptions

import tadkit.utils.param_spec as ps


# --- Test get_default_class_values --- #
def test_get_default_class_values():
    class A:
        def __init__(self, x=10, y="hello", z=None):
            pass

    defaults = ps.get_default_class_values(A)
    assert defaults == {"x": 10, "y": "hello", "z": None}


# --- Test get_param_descriptions --- #
def test_get_param_descriptions():
    class B:
        """
        Example class.

        Parameters
        ----------
        x : int, default=1
            The x parameter
        y : str
            The y parameter
        """

        def __init__(self, x=1, y="default"):
            pass

    desc = ps.get_param_descriptions(B)
    assert "x" in desc and desc["x"] == "The x parameter"
    assert "y" in desc and desc["y"] == "The y parameter"


# --- Test parse_sklearn_constraints --- #
def test_parse_sklearn_constraints_basic():
    constraints = {
        # Provide type, left, right, closed explicitly as per Interval signature
        "a": [Interval(Real, 0.0, 10.0, closed="both")],
        "b": ["integer", None],
        "c": [StrOptions({"red", "green"})],
        "d": [],  # no constraint
    }

    parsed = ps.parse_sklearn_constraints(constraints)

    # 'a' should map to float, with min=0, max=10
    assert parsed["a"]["type"] == float
    assert parsed["a"]["bounds"]["min"] == 0.0
    assert parsed["a"]["bounds"]["max"] == 10.0
    assert parsed["a"]["allow_none"] is False

    # 'b' integer + None → type int, allow_none True
    assert parsed["b"]["type"] == int
    assert parsed["b"]["allow_none"] is True

    # 'c' categorical string options
    assert parsed["c"]["type"] == "categorical"
    assert sorted(parsed["c"]["options"]) == ["green", "red"]
    assert parsed["c"]["allow_none"] is False

    # 'd' no constraints
    assert parsed["d"]["type"] is None
    assert parsed["d"]["allow_none"] is True


# --- Test anchor_type_to_default --- #
def test_anchor_type_to_default_behavior():
    entry = {
        "default": 5,
        "type": (int, float),
        "options": None,
        "allow_none": True,
    }
    anchored = ps.anchor_type_to_default(dict(entry))
    assert anchored["type"] is int
    # default is not None, so allow_none should become False
    assert anchored["allow_none"] is False

    entry_none = {
        "default": None,
        "type": (int, float),
        "options": ["a", "b"],
        "allow_none": False,
    }
    anchored_none = ps.anchor_type_to_default(dict(entry_none))
    assert anchored_none["type"] is type(None)
    assert anchored_none["options"] is None
    assert anchored_none["allow_none"] is True


# --- Test determine_widget --- #
@pytest.mark.parametrize(
    "entry,expected_widget",
    [
        (
            {
                "type": int,
                "default": 5,
                "bounds": {"min": 0, "max": 10},
                "allow_none": False,
                "options": None,
            },
            "slider",
        ),
        (
            {
                "type": float,
                "default": 2.5,
                "bounds": {"min": 0.0, "max": 10.0},
                "allow_none": False,
                "options": None,
            },
            "slider",
        ),
        (
            {
                "type": bool,
                "default": True,
                "bounds": {"min": None, "max": None},
                "allow_none": False,
                "options": None,
            },
            "checkbox",
        ),
        (
            {
                "type": str,
                "default": "abc",
                "bounds": {"min": None, "max": None},
                "allow_none": False,
                "options": None,
            },
            "text",
        ),
        (
            {
                "type": type(None),
                "default": None,
                "bounds": {"min": None, "max": None},
                "allow_none": True,
                "options": None,
            },
            "text",
        ),
        (
            {
                "type": "categorical",
                "default": "red",
                "bounds": {"min": None, "max": None},
                "allow_none": False,
                "options": ["red", "green"],
            },
            "select",
        ),
    ],
)
def test_determine_widget_various(entry, expected_widget):
    result = ps.determine_widget(dict(entry))
    assert result["widget"] == expected_widget
    assert "default" in result["widget_args"]


# --- Test params_from_class --- #
def test_params_from_class_combined():
    class C:
        """
        Sample class.

        Parameters
        ----------
        x : int, default=1
            x param
        y : str
            y param
        z : float
            z param
        """

        _parameter_constraints = {
            "x": [Interval(Integral, 0, 10, closed="both")],
            "y": [StrOptions({"a", "b"})],
            "z": ["number"],
        }

        def __init__(self, x=1, y="a", z=None):
            self.x = x
            self.y = y
            self.z = z

    spec = ps.params_from_class(C)
    # x should exist and widget slider (int bounds)
    assert "x" in spec
    assert spec["x"]["type"] is int
    assert spec["x"]["widget"] == "slider"
    # y should exist, categorical select
    assert spec["y"]["widget"] == "select"
    assert sorted(spec["y"]["options"]) == ["a", "b"]
    # z default is None, so type should reflect None or float, allow_none True
    assert spec["z"]["allow_none"] is True
    # Check description was populated
    assert spec["x"]["description"] == "x param"
    assert spec["y"]["description"] == "y param"
