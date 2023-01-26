# fmt: off
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

class WeightTrackingCallback(tf.keras.callbacks.Callback):
    """
    A callback to track weights during training.
    Weights can be tracked by keeping a direct history, checking sign flipping, etc...
    Each tracking method is implemented in a child class
    """

    def __init__(self, model: tf.keras.models.Model, reset_on_train_begin: bool = True, **kwargs):
        """
        Create a new instance of this call back.

        Parameters:
        model: tf.keras.models.Model
            The model to track weights of.
            Must be passed so lifecycle hooks can reference model!
        reset_on_train_begin: bool
            Flag to reset the callback on train begin - useful when changing tasks!
        """

        super().__init__()
        self.model = model
        # Okay, strap in
        # It appears that Tensorflow, in the backend, does NOT handle memory 
        # addresses correctly when it comes to models built from other models
        #
        # In particular, it appears that whatever model is "in use" overwrites 
        # any references to the previous model.
        # For our application, that means base_model (with, say, 2 layers) has 
        # 2 layers at init time, but that same reference will now point to
        # new_model (with, say, 4 layers) at callback time! For this reason,
        # A work around is to store a direct list of references to the model
        # layers to be used instead of self.model.layers
        # 
        # It appears this issue persists with other properties, i.e. 
        # the entire reference changes!
        self.model_layers = model.layers

        self.reset_on_train_begin = reset_on_train_begin