# fmt: off
from copy import deepcopy
from enum import Enum
from typing import List
import numpy as np
import os

from Utilities.Tasks.GenericTask import GenericTask
from .SignFlippingTracker import SignFlippingTracker
from .MomentumBasedTracker import MomentumBasedTracker
from .TotalWeightChangeTracker import TotalWeightChangeTracker
from .FisherInformationMatrixCalculator import FisherInformationMatrixCalculator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on


class EWC_Method(Enum):
    NONE = 1,
    WEIGHT_DECAY = 2,
    SIGN_FLIPPING = 3,
    MOMENTUM_BASED = 4,
    WEIGHT_CHANGE = 5,
    FISHER_MATRIX = 6,
    WEIGHT_MAGNITUDE = 7,
    INVERSE_WEIGHT_MAGNITUDE = 8,
    RANDOM = 9


class EWC_Term():
    """
    Representation of a single EWC_Term
    Collects together all EWC term ideas (lambda, optimal weights, omega...)
    Also exposes loss function for use

    Note the loss function only loops over the OPTIMAL weights given
    So if a larger omega matrix is given (e.g. Fisher over all weights) this is okay!
    The extra omega matrix is ignored
    """

    def __init__(self,
                 ewc_lambda: float,
                 optimal_weights: List[List[np.ndarray]],
                 omega_matrix: List[List[np.ndarray]]):
        """
        A single EWC term for model training

        Parameters:
            ewc_lambda: float
                The importance of this EWC term. 

            optimal_weights: List[List[np.ndarray]]
                The optimal weights of the model after training.
                Can be found by model.trainable_weights
                Note! Should only be the *shared* weights 

            omega_matrix: List[List[np.ndarray]]
                The weight importance matrix for this term.
                Should have the same dimensions (in every way) as 
                optimal_weights
        """

        self.ewc_lambda = ewc_lambda
        self.optimal_weights = deepcopy(optimal_weights)
        self.omega_matrix = deepcopy(omega_matrix)
        # max_omega = 0
        # # Scale omega matrix to have maximum of 1
        # for layer in self.omega_matrix:
        #     for weight in layer:
        #         weight_max = np.max(weight)
        #         max_omega = max(weight_max, max_omega)
        # # If condition prevents division by 0
        # if max_omega != 0:
        #     for layer_index in range(len(self.omega_matrix)):
        #         for weight_index in range(len(self.omega_matrix[layer_index])):
        #             self.omega_matrix[layer_index][weight_index] = \
        #                 self.omega_matrix[layer_index][weight_index] / max_omega
        

    def calculate_loss(self, model_layers: List[tf.keras.layers.Layer]):
        loss = 0
        for layer_index, layer in enumerate(model_layers):
            # Note in zip function - if omega is longer than optimal weights the excess omega is ignored
            # This may be an issue if omega and optimal weights are not in the same "position" of the network
            # TODO: Make EWC work better in the case of non-start of network shared weights...
            for omega, optimal, new in zip(self.omega_matrix[layer_index], self.optimal_weights[layer_index], layer.trainable_weights):
                loss += tf.reduce_sum(omega * tf.math.square(new-optimal))
        return loss * self.ewc_lambda/2


class EWC_Term_Creator():

    def __init__(self, ewc_method: EWC_Method, model: tf.keras.models.Model, callback_kwargs: dict = {}) -> None:
        """
        Initialize a new creator for EWC terms

        Parameters:
            method: EWC_Method
                Enum to set term creation method, e.g. sign_flip, weight_decay

            model: tf.keras.models.Model
                The model to calculate elastic weight consolidation
            
            callback_kwargs: dict
                A dictionary of any extra kwargs to pass to the callback
        """
        self.ewc_method = ewc_method
        self.model = model
        self.model_layers = model.layers
        self.callback_dict: dict[str, tf.keras.callbacks.Callback] = {}
        match self.ewc_method:
            case EWC_Method.SIGN_FLIPPING:
                self.callback_dict["SignFlip"] = SignFlippingTracker(model, **callback_kwargs)
            case EWC_Method.MOMENTUM_BASED:
                self.callback_dict["MomentumBased"] = MomentumBasedTracker(model, **callback_kwargs)
            case EWC_Method.WEIGHT_CHANGE:
                self.callback_dict["WeightChange"] = TotalWeightChangeTracker(model, **callback_kwargs)
            case EWC_Method.FISHER_MATRIX:
                self.fisher_calculator = FisherInformationMatrixCalculator()
            case _:
                pass

    def create_term(self, ewc_lambda: float, **kwargs) -> EWC_Term:
        """
        Create a new term using whatever method is specified and whatever
        data is collected at the call time. Should only be called at the
        end of a task!

        Parameters:
            ewc_lambda: float
                The importance of the new term
        """

        model_current_weights = []
        omega_matrix = []
        for layer_index, layer in enumerate(self.model_layers):
            model_layer_weights = []
            for weight_index, weight in enumerate(layer.trainable_weights):
                model_layer_weights.append(weight)
            model_current_weights.append(model_layer_weights)

        match self.ewc_method:
            case EWC_Method.NONE:
                for layer_index, layer in enumerate(self.model_layers):
                    current_omega = []
                    for weight_index, weight in enumerate(layer.trainable_weights):
                        current_omega.append(tf.zeros_like(weight))
                    omega_matrix.append(current_omega)
                return EWC_Term(ewc_lambda=0, optimal_weights=model_current_weights, omega_matrix=omega_matrix)

            case EWC_Method.WEIGHT_DECAY:
                # weight decay has omega as a matrix of 1's, which we can get from our already
                # calculated matrix of 0's !
                for layer_index, layer in enumerate(self.model_layers):
                    current_omega = []
                    for weight_index, weight in enumerate(layer.trainable_weights):
                        current_omega.append(tf.ones_like(weight))
                    omega_matrix.append(current_omega)
                return EWC_Term(ewc_lambda=ewc_lambda, optimal_weights=model_current_weights, omega_matrix=omega_matrix)

            case EWC_Method.SIGN_FLIPPING:
                sign_flip_callback: SignFlippingTracker = self.callback_dict["SignFlip"]  # type: ignore
                omega_matrix = []
                for layer_index, layer in enumerate(sign_flip_callback.sign_changes):
                    omega_layer = []
                    for weight_index, weight in enumerate(layer):
                        # An important weight flips *fewer* times
                        # So we take the reciprocal, which could
                        # Divide by zero, so also add 1
                        omega_layer.append(1/(1+weight))
                    omega_matrix.append(omega_layer)
                return EWC_Term(ewc_lambda=ewc_lambda, optimal_weights=model_current_weights, omega_matrix=omega_matrix)

            case EWC_Method.MOMENTUM_BASED:
                momentum_based_callback: MomentumBasedTracker = self.callback_dict["MomentumBased"]  # type: ignore
                omega_matrix = []
                for layer_index, layer in enumerate(momentum_based_callback.momenta_changes):
                    omega_layer = []
                    for weight_index, weight in enumerate(layer):
                        # An important weight changes direction *fewer* times (?)
                        # So we take the reciprocal, which could divide by zero
                        # so also add 1
                        omega_layer.append(1/(1+weight))
                    omega_matrix.append(omega_layer)
                return EWC_Term(ewc_lambda=ewc_lambda, optimal_weights=model_current_weights, omega_matrix=omega_matrix)

            case EWC_Method.WEIGHT_CHANGE:
                weight_change_callback: TotalWeightChangeTracker = self.callback_dict["WeightChange"]  # type: ignore
                omega_matrix = []
                for layer_index, layer in enumerate(weight_change_callback.total_distances):
                    omega_layer = []
                    for weight_index, weight in enumerate(layer):
                        # An important weight moves very little (?)
                        # So we take the reciprocal, which could divide by zero
                        # so also add 1
                        omega_layer.append(1/(1+weight))
                    omega_matrix.append(omega_layer)
                return EWC_Term(ewc_lambda=ewc_lambda, optimal_weights=model_current_weights, omega_matrix=omega_matrix)

            case EWC_Method.FISHER_MATRIX:
                # Calculate Fisher matrix and use as omega
                # To do this, need to run model over training dataset
                # And use this to approximate Fisher.
                #
                # Details here: https://towardsdatascience.com/an-intuitive-look-at-fisher-information-2720c40867d8
                # Example implementation: https://github.com/db434/EWC/blob/master/ewc.py
                # Original paper: https://arxiv.org/pdf/1612.00796.pdf
                # type: ignore

                try:
                    task = kwargs["task"]
                except KeyError:
                    print("ERROR: Fisher Matrix requires passing task as keyword 'task='")
                    exit(1)
                omega_matrix = self.fisher_calculator.generate_fisher_matrix(task)
                # type: ignore
                return EWC_Term(ewc_lambda=ewc_lambda, optimal_weights=model_current_weights, omega_matrix=omega_matrix)  # type: ignore

            case EWC_Method.WEIGHT_MAGNITUDE:
                for layer_index, layer in enumerate(self.model_layers):
                    current_omega = []
                    for weight_index, weight in enumerate(layer.trainable_weights):
                        current_omega.append(tf.identity(weight))
                    omega_matrix.append(current_omega)
                return EWC_Term(ewc_lambda=ewc_lambda, optimal_weights=model_current_weights, omega_matrix=omega_matrix)

            case EWC_Method.INVERSE_WEIGHT_MAGNITUDE:
                for layer_index, layer in enumerate(self.model_layers):
                    current_omega = []
                    for weight_index, weight in enumerate(layer.trainable_weights):
                        current_omega.append(tf.identity(tf.math.reciprocal(weight)))
                    omega_matrix.append(current_omega)
                return EWC_Term(ewc_lambda=ewc_lambda, optimal_weights=model_current_weights, omega_matrix=omega_matrix)

            case EWC_Method.RANDOM:
                for layer_index, layer in enumerate(self.model_layers):
                    current_omega = []
                    for weight_index, weight in enumerate(layer.trainable_weights):
                        current_omega.append(tf.random.uniform(shape=weight.shape, minval=0, maxval=1))
                    omega_matrix.append(current_omega)
                return EWC_Term(ewc_lambda=ewc_lambda, optimal_weights=model_current_weights, omega_matrix=omega_matrix)

            # Default case: return an empty term
            case _:
                return EWC_Term(ewc_lambda=0, optimal_weights=model_current_weights, omega_matrix=omega_matrix)

            


class EWC_Loss(tf.keras.losses.Loss):
    def __init__(self, base_loss: tf.keras.losses.Loss,
                 current_model_layers: List[tf.keras.layers.Layer],
                 EWC_terms: List[EWC_Term]):
        """
        Create a new instance of a loss function with EWC augmentation

        Parameters:
            base_loss: tf.keras.losses.Loss
                The base loss function (before EWC terms)
            current_model_layers: List[tf.keras.layers.Layer]
                The layers of the model to base the loss function off of
                The handles are needed to affect the EWC terms
                And the model itself cannot be used due to tensorflow memory management
            EWC_terms: List[EWC_Term]
                The additional EWC terms to augment loss with
        """

        super().__init__()
        self.base_loss = base_loss
        self.model_layers = current_model_layers
        self.ewc_terms = EWC_terms

    def call(self, y_true, y_pred):
        base_loss = self.base_loss(y_true, y_pred)
        ewc_loss = 0
        for term in self.ewc_terms:
            ewc_loss += term.calculate_loss(self.model_layers)
        return base_loss + ewc_loss
