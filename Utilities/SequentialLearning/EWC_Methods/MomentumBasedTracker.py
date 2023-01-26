# fmt: off
from enum import Enum
from .WeightTrackingCallback import WeightTrackingCallback
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on


class MomentumBasedTracker(WeightTrackingCallback):
    """
    Track the weights of a network by checking the direction of change of those weights after each epoch
    Basis of this tracker is to see how many times each weight changes "direction"
    This is a slightly more general version of sign flipping, working without requiring bipolar units
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
        # Track the number of times each weight changes momentum
        self.momenta_changes = []
        # Track the weights from the previous batch/epoch
        self.previous_weights = []
        # Track the momenta at the instant of the previous batch/epoch
        self.previous_momenta = []
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
            self.previous_momenta.append(zeros_like_weights)
            self.momenta_changes.append(zeros_like_weights)

    def measure_weight_changes(self):
        # Track the new weights and momenta to update at end
        updated_weights = []
        updated_momenta = []
        for layer_index, layer in enumerate(self.model_layers):
            layer_weights = []
            layer_momenta = []
            for weight_index, weight in enumerate(layer.trainable_weights):
                # Find the variables we need to calculate new momenta
                previous_weight = self.previous_weights[layer_index][weight_index]
                previous_momentum = self.previous_momenta[layer_index][weight_index]
                current_momentum = weight-previous_weight
                # Store new values for update at end
                layer_weights.append(tf.identity(weight))
                layer_momenta.append(tf.identity(current_momentum))

                # momenta changes defined by change of sign of momentum
                # So multiplying previous and current momentum is only -1 when sign changes
                self.momenta_changes[layer_index][weight_index] += \
                    tf.cast(tf.sign(current_momentum*previous_momentum) == -1, dtype=tf.float32)
            updated_weights.append(layer_weights)
            updated_momenta.append(layer_momenta)
        self.previous_weights = updated_weights
        self.previous_momenta = updated_momenta

    def on_train_begin(self, logs=None):
        if self.reset_on_train_begin:
            self.reset_tracking()

    def on_batch_end(self, epoch, logs=None):
        if self.measure_period == self.MeasurePeriod.BATCH_END:
            self.measure_weight_changes()

    def on_epoch_end(self, epoch, logs=None):
        if self.measure_period == self.MeasurePeriod.EPOCH_END:
            self.measure_weight_changes()