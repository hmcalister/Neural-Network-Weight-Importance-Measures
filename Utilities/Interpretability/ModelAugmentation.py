# fmt: off
from copy import deepcopy
from enum import Enum
from typing import Callable, List, Sequence
import numpy as np

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

class ComparisonMethod(Enum):
    MODEL_WISE = 0,
    LAYER_WISE = 1,

class AggregationLevel(Enum):
    NO_AGGREGATION = 0,
    UNIT = 1,
    CONV_FILTER = 2,

class AggregationMethod(Enum):
    MEAN_AVERAGE = 0,
    MINIMUM = 1,
    MAXIMUM = 2,


def _aggregate(matrices: Sequence[np.ndarray], aggregation_method: AggregationMethod):
    """
    Aggregate a list of tensors together to a single value
    Returns a list of numpy arrays with the same shape but filled with the aggregated value
    """

    aggregated_matrices = []
    for m in matrices:
        if m.shape==[]:
            aggregated_matrices.append(m)
            continue
        match aggregation_method:
            case AggregationMethod.MEAN_AVERAGE:
                aggregation_value = tf.reduce_mean(m, axis=-1)
            case AggregationMethod.MINIMUM:
                aggregation_value = tf.reduce_min(m, axis=-1)
            case AggregationMethod.MAXIMUM:
                aggregation_value = tf.reduce_max(m, axis=-1)
            case _:
                aggregation_value = m
        aggregation_value = tf.expand_dims(aggregation_value, axis=-1)
        aggregated_matrices.append(np.full_like(m, aggregation_value))
    return aggregated_matrices

def aggregate_omega(
    omega_matrix: List[List[tf.Variable]],
    base_model: tf.keras.models.Model,
    aggregation_level: AggregationLevel,
    aggregation_method: AggregationMethod,
    ) -> List[List[tf.Variable]]:
    """
    Take a weight importance matrix and return an aggregated matrix with the same dimensionality
    (except possibly the final dimension which may be reduced)
    Should not impact loop definitions etc
    """

    match aggregation_level:
        case AggregationLevel.NO_AGGREGATION:
            return omega_matrix

        case AggregationLevel.UNIT:
            aggregated_omega = []

            for layer_index, layer in enumerate(base_model.layers):
                omega_layer = omega_matrix[layer_index]
                aggregated_layer = _aggregate(omega_layer, aggregation_method) # type: ignore      
                aggregated_omega.append(aggregated_layer)
            return aggregated_omega

        case AggregationLevel.CONV_FILTER:
            # Much like UNIT aggregation but this time we only aggregate over conv filters
            # A conv filter can be checked by looking at the layer instance (conv2d) 
            # and taking the first set of weights
            aggregated_omega = []

            for layer_index, layer in enumerate(base_model.layers):
                omega_layer = omega_matrix[layer_index]
                aggregated_layer = []
                if not isinstance(layer, tf.keras.layers.Conv2D):
                    aggregated_layer = omega_layer
                else:
                    # Aggregate only the first weight: much easier than entire unit!
                    aggregated_filter = _aggregate([omega_layer[0]], aggregation_method) # type: ignore                    
                    aggregated_layer = [aggregated_filter[0], omega_layer[1]]
                aggregated_omega.append(aggregated_layer)
            return aggregated_omega


def threshold_model_by_omega(
    base_model: tf.keras.models.Model,
    omega_matrix: List[List[np.ndarray]],
    threshold_percentage: float,
    comparison_method: ComparisonMethod,
    aggregation_level: AggregationLevel = AggregationLevel.NO_AGGREGATION,
    aggregation_method: AggregationMethod = AggregationMethod.MEAN_AVERAGE,
    verbose: bool = False
    ) -> tf.keras.models.Model:
    """
    Given a base model and some measure of weight importance, create a new model
    that has the same weights as the base model for the most important weights and
    zero for non-important weights

    Implementation first finds the distribution of weights, then determines which 
    weights are above the threshold value supplied (as a proportion) and zeros weights
    with importance below the threshold

    Parameters:
        base_model: tf.keras.models.Model
            The model to augment, taking weights from
            The model is copied, so the original model is unaffected by this operation
        omega_matrix: List[List[np.ndarray]]
            The measure of weight importance to apply to the base model
            Note this requires the shapes of this matrix and model weights be the same
        threshold_percentage: float
            The threshold of importance to keep an associated weight
            Given as a float between 0 (keep all weights) and 1 (keep no weights)
            Note this is percentage i.e. linear. Not a standard deviation or Z score
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
        verbose: bool
            Flag to print the measurements and weight values during thresholding
    """

    new_model = tf.keras.models.clone_model(base_model)
    threshold_value: np.float32 = np.float32(0)
    
    omega_matrix = aggregate_omega(omega_matrix, base_model, aggregation_level, aggregation_method)  # type: ignore

    # If we are comparing across the entire model, do this before thresholds
    if comparison_method == ComparisonMethod.MODEL_WISE:
        flat_omega = []  # type: ignore
        for layer_index, layer in enumerate(omega_matrix):
            for weight_index, omega in enumerate(layer):
                flat_omega = tf.concat([flat_omega, tf.reshape(omega, [-1])], axis=0)
        flat_omega = tf.sort(flat_omega)
        threshold_index = min(int(len(flat_omega) * threshold_percentage), len(flat_omega)-1)
        threshold_value = flat_omega[threshold_index].numpy()
        if verbose:
            print(f"MODEL_WISE {threshold_value=}")

    for layer_index, (omega_layer, model_layer) in enumerate(zip(omega_matrix, base_model.layers)):
        if comparison_method == ComparisonMethod.LAYER_WISE:
            flat_omega = []  # type: ignore
            for weight_index, (omega, _) in enumerate(zip(omega_layer, model_layer.trainable_weights)):
                flat_omega = tf.concat([flat_omega, tf.reshape(omega, [-1])], axis=0)
            flat_omega = tf.sort(flat_omega)
            if len(flat_omega)==0:
                # This is a strange condition
                # In short, Input layers have NO weights, nada, none, []
                # So we cannot even index into them with threshold_index = 0
                # but we still need to run the remainder of this loop to 
                # correctly add the empty array to the new model, so
                # instead we just hack a value to prevent a crash and move on
                threshold_value = np.float32(0)
            else:
                # In case we specified 1.0 thresholding, don't run off the end!
                threshold_index = min(int(len(flat_omega) * threshold_percentage), len(flat_omega)-1)
                threshold_value = flat_omega[threshold_index].numpy()
                if verbose:
                    print(f"LAYER_WISE {layer_index=} {model_layer.name} {threshold_value=}")

        new_layer_weights = []
        omega_index = 0
        for weight_index, model_weight in enumerate(model_layer.weights):
            new_weight = deepcopy(model_weight.numpy())
            if model_weight.trainable:
                omega = omega_layer[omega_index]
                omega_index+=1
                replacement_array = np.random.normal(0, np.std(model_weight)/10, model_weight.shape)
                # replacement_array = np.zeros_like(new_weight)
                new_weight[omega < threshold_value] = replacement_array[omega < threshold_value]
            new_layer_weights.append(new_weight)
        new_model.layers[layer_index].set_weights(new_layer_weights)

    # Yeah, I'm setting a private field, sue me
    new_model._name = f"{threshold_percentage}-threshold_{base_model.name}"
    optimizer = deepcopy(base_model.optimizer)
    loss_fn = deepcopy(base_model.loss)
    run_eagerly = base_model.run_eagerly
    new_model.compile(
        optimizer=optimizer,
        loss = loss_fn,
        run_eagerly=run_eagerly
    )

    return new_model