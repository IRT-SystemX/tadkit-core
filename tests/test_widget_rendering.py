import json
from tadkit.utils.ui import WidgetFactory

param = {
    "type": float,
    "bounds": {"min": 0, "max": 1, "closed": "right"},
    "allow_none": True,
    "default": None,
    "description": "Proportion of data to use for robust covariance estimation.",
}

factory = WidgetFactory("no_ui")
numeric_spec = factory.make_numeric(
    "Alpha",
    param["default"],
    0,
    1,
    float,
    param["description"],
    "right",
    param["allow_none"],
)
print(json.dumps(numeric_spec, indent=2))
