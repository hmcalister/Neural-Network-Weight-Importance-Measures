# fmt: off

from Utilities.Tasks.GenericTask import GenericTask
from .EWC_Methods.EWC_Methods import *
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, List, TextIO, Union


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

class SequentialLearningManager():
    """
    A manager for sequential learning environments. Takes a model
    and a list of tasks to train on (ordered). Tasks must be an object
    (SequentialTask) that offers a task head (placed at end of model) the
    task data (tf.keras.Dataset) and some validation data to be tested each epoch.
    """

    MODEL_SAVE_BASE_PATH = "models/sequential_models"

    class SequentialValidationCallback(tf.keras.callbacks.Callback):
        def __init__(self, tasks: List[GenericTask]):
            """
            Create a new validation callback, checking model performance 
            on all task each epoch

            Parameters:
                tasks: List(SequentialTask):
                    The list of tasks to learn on (ordered in a list).
                    See SequentialTask class for more details
            """
            super().__init__()
            self.tasks = tasks
            self.validation_results = {}

        def on_epoch_end(self, epoch, logs=None):
            epoch_results = {}
            print(f"\nVALIDATING AFTER EPOCH {epoch+1}")
            for task_index in range(len(self.tasks)):
                task_results: dict = self.tasks[task_index].evaluate_model()
                for k,v in task_results.items():
                    if k not in epoch_results:
                        epoch_results[k] = [v]
                    else:
                        epoch_results[k].append(v)
            for k,v in epoch_results.items():
                if k in self.validation_results:
                    self.validation_results[k].append(v)
                else:
                    self.validation_results[k] = [v]
            print(f"FINISHED VALIDATION")

    def __init__(self, base_model: tf.keras.models.Model,
                 tasks: List[GenericTask],
                 epochs: Union[int, List[int]],
                 EWC_method: EWC_Method = EWC_Method.NONE,
                 ewc_lambda: float = 0.0,
                 log_file_path="logs/Manager.log",):
        """
        Create a new SequentialLearningManager

        Parameters:
            :param base_model tf.keras.models.Model:
                The base model (or more accurately, a model with all shared weights)
            :param tasks List(SequentialTask):
                The list of tasks to learn on (ordered in a list).
                See SequentialTask class for more details
            :param EWC_method:
                Method to calculate elastic weight consolidation importance's
                Presence also adds EWC term to subsequent loss functions
            :param ewc_lambda float:
                The value of lambda to use for EWC terms
            :param log_file_path String:
                The path to the log file to be used
                Defaults to logs/Manager.log
        """

        self.base_model:tf.keras.models.Model = base_model
        self.base_model_layers: List[tf.keras.layers.Layer] = [l for l in base_model.layers]
        self.tasks: List[GenericTask] = tasks
        self.EWC_terms: List[EWC_Term] = []
        self.training_histories: List[tf.keras.callbacks.History] = []
        self._current_task_index: int = 0
        self.EWC_term_creator = EWC_Term_Creator(EWC_method, base_model)
        self.ewc_lambda = ewc_lambda

        self.validation_callback = \
            SequentialLearningManager.SequentialValidationCallback(tasks)

        # We're doing a little hack here so type hinting is thrown off
        # After this block, epochs is type List[int]
        self.epochs = epochs  # type: ignore
        if isinstance(epochs, int):
            self.epochs: List[int] = [epochs for _ in range(len(self.tasks))]

        self._log_file: TextIO = open(log_file_path, "wt")

    def train_all(self, callbacks: List[tf.keras.callbacks.Callback] = []):
        """
        Train all tasks, sequentially

        Parameters:
            callbacks: List[tf.keras.callbacks.Callback]
                The callbacks to add to each task
        """

        while self._current_task_index < len(self.tasks):
            print(f"---***--- {self.tasks[self._current_task_index].name} ---***---")
            self.train_next_task(callbacks)
        
    def train_next_task(self, callbacks: List[tf.keras.callbacks.Callback] = []):
        """
        Begin training on the next task in the list, or return None if no such task exists

        Parameters:
            callbacks: List[tf.keras.callbacks.Callback]
                The callbacks to add to this task
        """

        if self._current_task_index >= len(self.tasks):
            return None

        current_task = self.tasks[self._current_task_index]

        # Recompile model to use new loss
        # Note we keep the metrics!
        for task in self.tasks:
            base_loss_function = task.model_base_loss
            task.compile_model(EWC_Loss(base_loss_function, self.base_model_layers, self.EWC_terms))

        # Train model, store history
        history = current_task.train_on_task(epochs=self.epochs[self._current_task_index],
                    callbacks=[self.validation_callback, *self.EWC_term_creator.callback_dict.values(), *callbacks])
        self.training_histories.append(history)

        # Save this model to disk after training
        # Notice we use model name here, possible security risk/bug if model name isn't valid path!
        current_task.model.save(SequentialLearningManager.MODEL_SAVE_BASE_PATH+f"/{current_task.model.name}")

        # Create an EWC term for the now completed task
        self.EWC_terms.append(self.EWC_term_creator.create_term(ewc_lambda=self.ewc_lambda, task=current_task))
        self._current_task_index += 1

    def get_training_histories(self):
        return self.training_histories

    def get_validation_data(self):
        return self.validation_callback.validation_results

    def plot_validation_callback_data(self, key:str, title: str="", ylabel: str=""):
        """
        Plot the data from the validation callback
        Possible keys are any metric name or 'loss', e.g.
        key is in the set {"loss", "base_loss", "val_loss", "val_base_loss"} 
        """

        data = self.validation_callback.validation_results[key]
        marker=None
        markersize=5
        if len(data) < 100:
            marker = 'o'

        fig = plt.figure(figsize=(12,8))
        plt.plot(data, marker=marker, markersize=markersize)
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.legend([t.name for t in self.tasks])
        task_boundaries = np.cumsum(self.epochs)-1
        plt.vlines(task_boundaries, colors="k",  # type: ignore
            ymin=np.min(data),
            ymax=np.max(data),  # type: ignore
            linestyles="dashed", alpha=0.5)
        plt.tight_layout()
        plt.show()

    def log_to_file(self, log_string: Any):
        """
        A quick and overly simple logging function to dump something to a file
        Separates something potentially interesting from screens of TF logs

        Parameters:
            log_string: str
                The string to append to the log file
        """

        self._log_file.write(log_string)
        self._log_file.write("\n")