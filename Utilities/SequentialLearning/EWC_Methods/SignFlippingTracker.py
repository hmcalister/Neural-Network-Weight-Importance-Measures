# fmt: off
from enum import Enum
from .WeightTrackingCallback import WeightTrackingCallback
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on


class SignFlippingTracker(WeightTrackingCallback):
    """
    Track the weights of a network by checking the sign of those weights after each epoch
    Count how often each weight flips sign
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
        self.sign_changes = []
        self.stored_weights = []
        self.reset_tracking()

    def reset_tracking(self):
        for layer in self.model_layers:
            layer_weights = []
            layer_sign_change = []
            for weight in layer.trainable_weights:
                layer_weights.append(weight)
                layer_sign_change.append(tf.zeros_like(weight))
            self.stored_weights.append(layer_weights)
            self.sign_changes.append(layer_sign_change)

    def measure_weight_changes(self):
        updated_weights = []
        for layer_index, layer in enumerate(self.model_layers):
            layer_weights = []
            curr_layer = self.stored_weights[layer_index]
            for weight_index, weight in enumerate(layer.trainable_weights):
                layer_weights.append(tf.identity(weight))
                sign_change = tf.sign(weight*curr_layer[weight_index]) == -1
                self.sign_changes[layer_index][weight_index] += tf.cast(sign_change, dtype=tf.float32)
            updated_weights.append(layer_weights)
        self.stored_weights = updated_weights

    def on_train_begin(self, logs=None):
        if self.reset_on_train_begin:
            self.reset_tracking()

    def on_batch_end(self, epoch, logs=None):
        if self.measure_period == self.MeasurePeriod.BATCH_END:
            self.measure_weight_changes()

    def on_epoch_end(self, epoch, logs=None):
        if self.measure_period == self.MeasurePeriod.EPOCH_END:
            self.measure_weight_changes()
