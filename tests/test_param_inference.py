from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from tadkit.utils.param_spec import params_from_class
from tadkit.utils.render_widgets_from_params import render_widgets_from_params

params_from_class(KernelDensity)

gmm_spec = params_from_class(GaussianMixture)

widgets, value_fun = render_widgets_from_params(gmm_spec, frontend="ipywidgets")
value_fun()
