# Changelog


## v 1.2.0 (2025-12-31, current)
* Formatter (replacing Formalizer) array-agnostic (pandas & numpy) interface for ML pipelines; default RawToWideFormatter is provided.
* Registry singleton for dynamically tracking and matching learner classes; base learners (Kernel density-based, Gaussian mixtures ...) implemented from sklearn.
* Simplify tadlearner protocol interface (no more properties, only methods) and remove old wrappers to Confiance.ai (tdaad and kcpdi fully supported and functional).
* UPD decomposable_tadlearner_factory and remove tadlearner_factory (useless now)
* Visualiser streamlit various bug fixes and misc.

## v 1.1.1 (2025-10-23)
* clean-up for ease of use / learning
* visualiser streamlit for demonstration / embedding

## v 1.1.0 (2025-03-31)
* Open release of tadkit-core



