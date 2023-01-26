# fmt: off
from enum import Enum
from .WeightTrackingCallback import WeightTrackingCallback
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on


class TotalWeightChangeTracker(WeightTrackingCallback):
    """
    Calculate weight importance using the total distance the weight has "moved" each epoch
    """

    class MeasurePeriod(Enum):
        BATCH_END = 1,
        EPOCH_END = 2

    def __init__(self, model: tf.keras.models.Model,
                 measure_period: MeasurePeriod = MeasurePeriod.BATCH_END,
                 **kwargs):
        """
        Create a new instance of this call back.

        Parameters:
            model: tf.keras.models.Model
                The model to track weights of.
                Must be passed so lifecycle hooks can reference model!

            measure_period: MeasurePeriod
                How often to measure the weights for updates
        """

        super().__init__(model, **kwargs)
        self.measure_period = measure_period
        self.total_distances = []
        # Track the weights from the previous batch/epoch
        self.previous_weights = []
        self.reset_tracking()

    def reset_tracking(self):
        for layer in self.model_layers:
            layer_weights = []
            zeros_like_weights = []
            for weight in layer.trainable_weights:
                layer_weights.append(weight)
                zeros_like_weights.append(tf.zeros_like(weight))
            # Use exact value of weight for previous weight
            self.previous_weights.append(layer_weights)
            # We do not have information on derivatives for single instance
            # So default these to zeros
            self.total_distances.append(zeros_like_weights)

    def measure_weight_changes(self):
        # Track the new weights to update at end
        updated_weights = []
        for layer_index, layer in enumerate(self.model_layers):
            layer_weights = []
            for weight_index, weight in enumerate(layer.trainable_weights):
                # Find the variables we need to calculate new distance traveled
                previous_weight = self.previous_weights[layer_index][weight_index]
                # Store new values for update at end
                layer_weights.append(tf.identity(weight))
                self.total_distances[layer_index][weight_index] += weight-previous_weight
            updated_weights.append(layer_weights)
        self.previous_weights = updated_weights

    def on_train_begin(self, logs=None):
        if self.reset_on_train_begin:
            self.reset_tracking()

    def on_batch_end(self, epoch, logs=None):
        if self.measure_period == self.MeasurePeriod.BATCH_END:
            self.measure_weight_changes()

    def on_epoch_end(self, epoch, logs=None):
        if self.measure_period == self.MeasurePeriod.EPOCH_END:
            self.measure_weight_changes()