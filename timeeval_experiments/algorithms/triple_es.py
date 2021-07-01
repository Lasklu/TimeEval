from durations import Duration
from sklearn.model_selection import ParameterGrid
from typing import Any, Optional

from timeeval import Algorithm
from timeeval.adapters import DockerAdapter
from timeeval.data_types import TrainingType, InputDimensionality


_triple_es_parameters = {
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 },
 "seasonal": {
  "defaultValue": "add",
  "description": "type of seasonal component",
  "name": "seasonal",
  "type": "enum[add, mul]"
 },
 "seasonal_periods": {
  "defaultValue": 100,
  "description": "number of time units at which events happen regularly/periodically",
  "name": "seasonal_periods",
  "type": "int"
 },
 "train_window_size": {
  "defaultValue": 200,
  "description": "size of each TripleES model to predict the next timestep",
  "name": "train_window_size",
  "type": "int"
 },
 "trend": {
  "defaultValue": "add",
  "description": "type of trend component",
  "name": "trend",
  "type": "enum[add, mul]"
 }
}


def triple_es(params: Any = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="TripleES",
        main=DockerAdapter(
            image_name="mut:5000/akita/triple_es",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        params=_triple_es_parameters,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
