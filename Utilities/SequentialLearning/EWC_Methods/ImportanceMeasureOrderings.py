# fmt: off
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Utilities.Tasks.GenericTask import GenericTask as Task
from Utilities.Interpretability.ModelAugmentation import threshold_model_by_omega, ComparisonMethod, AggregationLevel, AggregationMethod

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

def validation_loss_by_importance_threshold(
        task: Task,
        omega_matrix: List[List[np.ndarray]],
        threshold_values: List[float],
        show_plot: bool = True,
        comparison_method: ComparisonMethod = ComparisonMethod.LAYER_WISE,
        aggregation_level: AggregationLevel = AggregationLevel.NO_AGGREGATION,
        aggregation_method: AggregationMethod = AggregationMethod.MEAN_AVERAGE,
    ) -> pd.DataFrame:
    """
    Given a task (importantly, a model with loss and a validation dataset - should signature be changed??)
    calculate the validation loss of that model with different levels of threshold

    Parameters:
        task: GenericTask
            The task to take the model, loss, and validation dataset from
            In future, signature may change to take these values more explicitly, but for now
            the validation dataset will almost certainly be created from a task anyway!

        omega_matrix: List[List[np.ndarray]]
            The weight importance measures corresponding to the model, to be used for thresholds

        threshold_values: List[float]
            The threshold values to check the validation loss of. This list will be sorted before operation

        show_plot: bool
            Flag to plot the data from this trial. Defaults to True

        comparison_method: ComparisonMethod
            Method for determining actual threshold values i.e. compare weights 
            only within a layer (LAYER_WISE) or across the entire model (MODEL_WISE)

        aggregation_level: AggregationLevel(Enum):
            Enum to select how to handle aggregation of weight importances during 
            threshold. NO_AGGREGATION (default) does not combine weight importances
            and considers each weight (with importance) individually. UNIT method
            considers each unit at a time, aggregating the unit weight importances.
            This means entire units are either kept or not, never partial units.

        aggregation_method: AggregationMethod(Enum):
            If aggregation_level is not NO_AGGREGATION this parameter determines how
            to perform the aggregation. Options include mean average, minimum, maximum
    """

    threshold_values.sort()
    base_model = task.model
    loss_fn = task.model_base_loss
    training_dataset = task.training_dataset
    training_batches = task.training_batches
    validation_dataset = task.validation_dataset
    validation_batches = task.validation_batches
    column_names = ["Threshold Value", "Loss", "Validation Loss"]
    results_df = pd.DataFrame(columns=column_names)

    for threshold_value in threshold_values:
        print(f"{threshold_value=}{' '*80}", end="\r")
        model = threshold_model_by_omega(base_model, omega_matrix, threshold_value, comparison_method, aggregation_level, aggregation_method)
        model.compile(loss=loss_fn)
        loss = model.evaluate(training_dataset, steps=training_batches, verbose=0) # type: ignore
        val_loss = model.evaluate(validation_dataset, steps=validation_batches, verbose=0) # type: ignore
        results_df = pd.concat([results_df, pd.DataFrame([[threshold_value, loss, val_loss]], columns=column_names)], ignore_index=True, axis=0)

    if show_plot:
        plt.plot(results_df["Threshold Value"], results_df["Loss"], marker="x", label="Loss")
        plt.plot(results_df["Threshold Value"], results_df["Validation Loss"], marker="x", label="Val Loss")
        plt.xlabel("Threshold Value")
        plt.ylabel("Loss")
        plt.title("Validation Loss over Weight Importance Threshold Values")
        plt.legend()
        plt.show()

    return results_df
