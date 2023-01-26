# fmt: off
from copy import deepcopy
import os
from typing import List
from Utilities.Tasks.GenericTask import GenericTask

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

CLIPPED_STANDARD_DEVIATIONS = 3

class FisherInformationMatrixCalculator():
    """
    A callback to track tasks and calculate the fisher information matrix when called
    This class probably doesn't need to be a callback, but it also doesn't hurt if it's needed
    """

    def __init__(self, samples: int=-1):
        """
        Create a new fisher information callback - creating a new fisher information matrix
        at the end of each task

        Note! This callback will track gradients for the ENTIRE network,
        not only the "shared" section. See the EWC_Term class for why this is not a problem

        Parameters:
            tasks: A list of sequential tasks 
            samples: The number of samples to take from the training data to construct the matrix
                Defaults to -1, meaning take the entire dataset
        """
        super().__init__()
        self.samples: int = samples
        self.fisher_matrices: List[List[List[tf.Tensor]]] = []

    def generate_fisher_matrix(self, task: GenericTask):
        # Do fisher calculation
        samples = self.samples if self.samples!=-1 else task.training_batches

        # Notice for this section we work with flattened arrays until the end (for speed)
        # init variance to be a zero matrix like all weight tensors of task model
        variance = [tf.zeros_like(tensor) for tensor in task.model.trainable_weights]
        # Noting gradients, apply model to samples
        step = 0
        # Each loop we get the gradients for a new sample
        # And update the variance to account for the new sample
        for x, _ in task.training_dataset.take(samples):
            outputs = []
            # Track the gradient over the log likelihood
            with tf.GradientTape() as tape:
                outputs = task.model(x)
                log_likelihood = tf.math.log(outputs)
            # Finally actually take the gradient and update the variance accordingly
            gradients = tape.gradient(log_likelihood, task.model.trainable_weights)
            variance = [var + (grad ** 2) for var, grad in zip(variance, gradients)] # type: ignore            
            step+=1
        # Finally - Fisher matrix is found by variance divided by number of samples
        fisher_diagonal = [tensor / samples for tensor in variance]

        # Here we clip the top few percentile of the weights to avoid massive gradient explosions
        # flat_fisher = tf.concat([tf.reshape(l, [-1]) for l in fisher_diagonal], axis=0)
        # fisher_mean = tf.math.reduce_mean(flat_fisher)
        # fisher_std = tf.math.reduce_std(flat_fisher)
        # upper_limit = fisher_mean + CLIPPED_STANDARD_DEVIATIONS * fisher_std

        # Now we can stop working with flat arrays and convert to layer-collected format
        # But first, convert with these ugly loops
        current_fisher = []
        fisher_index = 0
        for layer_index, layer in enumerate(task.model.layers):
            current_layer = []
            for tensor_index, tensor in enumerate(layer.trainable_weights):
                # f = tf.clip_by_value(tensor, 0, upper_limit)
                current_layer.append(deepcopy(fisher_diagonal[fisher_index]))
                fisher_index += 1
            current_fisher.append(current_layer)
        self.fisher_matrices.append(current_fisher)
        return current_fisher
