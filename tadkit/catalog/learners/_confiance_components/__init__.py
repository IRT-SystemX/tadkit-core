from ....utils import NotImportedWrapper
from ._cnndrad_wrapper import get_wrapped_datareconstructionad
from ._kcpdi_wrapper import get_wrapped_kcplearner
from ._sbad_wrapper import get_wrapped_dilanodetectm
from ._tdaad_wrapper import get_wrapped_topolad_pp

try:
    DataReconstructionADLearner = get_wrapped_datareconstructionad()
except ModuleNotFoundError as exc:
    DataReconstructionADLearner = NotImportedWrapper(exc)

try:
    KcpLearner = get_wrapped_kcplearner()
except ModuleNotFoundError as exc:
    KcpLearner = NotImportedWrapper(exc)

try:
    DiLAnoDetectmLearner = get_wrapped_dilanodetectm()
except ModuleNotFoundError as exc:
    DiLAnoDetectmLearner = NotImportedWrapper(exc)

try:
    TopologicalAnomalyDetector = get_wrapped_topolad_pp()
except ModuleNotFoundError as exc:
    TopologicalAnomalyDetector = NotImportedWrapper(exc)
