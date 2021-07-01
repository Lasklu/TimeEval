from durations import Duration
from sklearn.model_selection import ParameterGrid
from typing import Any, Optional

from timeeval import Algorithm
from timeeval.adapters import DockerAdapter
from timeeval.data_types import TrainingType, InputDimensionality


_fft_parameters = {
 "context_window_size": {
  "defaultValue": 21,
  "description": "Centered window of neighbors to consider for the calculation of local outliers' z_scores",
  "name": "context_window_size",
  "type": "int"
 },
 "fft_parameters": {
  "defaultValue": 5,
  "description": "Number of parameters to be used in IFFT for creating the fit.",
  "name": "fft_parameters",
  "type": "int"
 },
 "local_outlier_threshold": {
  "defaultValue": 0.6,
  "description": "Outlier threshold in multiples of sigma for local outliers",
  "name": "local_outlier_threshold",
  "type": "float"
 },
 "max_anomaly_window_size": {
  "defaultValue": 50,
  "description": "Maximum size of outlier regions.",
  "name": "max_anomaly_window_size",
  "type": "int"
 },
 "max_sign_change_distance": {
  "defaultValue": 10,
  "description": "Maximum gap between two closed oppositely signed local outliers to detect a sign change for outlier region grouping.",
  "name": "max_sign_change_distance",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for the random number generator",
  "name": "random_state",
  "type": "int"
 }
}


def fft(params: Any = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="FFT",
        main=DockerAdapter(
            image_name="mut:5000/akita/fft",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        params=_fft_parameters,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
