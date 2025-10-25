from sklearn.linear_model import LogisticRegression
from tadkit.utils.param_spec import params_from_class
from tadkit.utils.ui import render_widgets_from_params


params = params_from_class(LogisticRegression)

widgets = render_widgets_from_params(params, frontend="ipywidgets")
widgets["n_jobs"].value
