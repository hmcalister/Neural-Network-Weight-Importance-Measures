# %load_ext tensorboard
# fmt: off
from Utilities.Interpretability.ModelAugmentation import *
from Utilities.Interpretability.InterpretabilityMethods import *
from Utilities.Tasks.MNISTClassificationTask import MNISTClassificationTask as Task
from Utilities.SequentialLearning.EWC_Methods.EWC_Methods import *
from Utilities.SequentialLearning.EWC_Methods.ImportanceMeasureOrderings import validation_loss_by_importance_threshold
from Utilities.Interpretability.ModelAugmentation import ComparisonMethod, AggregationLevel, AggregationMethod
import pandas as pd
import matplotlib.pyplot as plt
import sys


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

if len(sys.argv)<=1:
    print("NO ARG GIVEN: RUN SCRIPT LIKE `./main.py [TRIAL_ID]` TO ASSIGN A TRIAL ID")
    exit()

TRIAL_ID = sys.argv[1]
print(f"TRIAL {TRIAL_ID}: {tf.config.list_physical_devices('GPU')}")

MODEL_SAVE_PATH = f"models/MNIST_parallel_model_{TRIAL_ID}/"
RUN_EAGERLY = False

image_size = Task.IMAGE_SIZE
task_labels = [0,1,2,3,4,5,6,7,8,9]
model_input_shape = image_size
batch_size = 32

model_inputs = model_layer = tf.keras.Input(shape=model_input_shape)
model_layer = tf.keras.layers.Conv2D(32, (3,3), activation="relu", name="conv2d_0")(model_layer)
model_layer = tf.keras.layers.Conv2D(32, (3,3), activation="relu", name="conv2d_1")(model_layer)
model_layer = tf.keras.layers.BatchNormalization()(model_layer)
model_layer = tf.keras.layers.MaxPool2D((2,2))(model_layer)
model_layer = tf.keras.layers.Conv2D(64, (3,3), activation="relu", name="conv2d_2")(model_layer)
model_layer = tf.keras.layers.Conv2D(64, (3,3), activation="relu", name="conv2d_3")(model_layer)
model_layer = tf.keras.layers.BatchNormalization()(model_layer)
model_layer = tf.keras.layers.MaxPool2D((2,2))(model_layer)
model_layer = tf.keras.layers.Flatten()(model_layer)
model_layer = tf.keras.layers.Dense(64, activation="relu")(model_layer)
model_layer = tf.keras.layers.Dropout(0.2)(model_layer)
model_layer = tf.keras.layers.Dense(64, activation="relu")(model_layer)
model_layer = tf.keras.layers.Dropout(0.2)(model_layer)
model_layer = tf.keras.layers.Dense(len(task_labels))(model_layer)
model = tf.keras.Model(inputs=model_inputs, outputs=model_layer, name="base_model")
model.summary()

if len(task_labels) == 2:
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
else:
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

training_image_augmentation = None
training_image_augmentation = tf.keras.Sequential([
    # tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomZoom(
            height_factor=(-0.05, -0.25),
            width_factor=(-0.05, -0.25)),
    tf.keras.layers.RandomRotation(0.05, "constant")
])

task = Task(
        name=f"Task 0",
        model=model,
        model_base_loss=loss_fn,
        task_labels=task_labels,
        training_batches = 0,
        validation_batches = 0,
        batch_size=batch_size,
        training_image_augmentation = training_image_augmentation,
        run_eagerly=RUN_EAGERLY
    )

ewc_methods = [
    EWC_Method.FISHER_MATRIX,
    EWC_Method.SIGN_FLIPPING,
    EWC_Method.MOMENTUM_BASED,
    EWC_Method.WEIGHT_CHANGE,
    # EWC_Method.WEIGHT_MAGNITUDE,
    # EWC_Method.INVERSE_WEIGHT_MAGNITUDE,
    EWC_Method.RANDOM,
]

aggregation_levels = [
    AggregationLevel.NO_AGGREGATION,
    AggregationLevel.UNIT,
    AggregationLevel.CONV_FILTER,
]

ewc_term_creators = [EWC_Term_Creator(ewc_method, model, callback_kwargs={"reset_on_train_begin": False}) for ewc_method in ewc_methods]

# Add all callbacks from all terms to the callback list
callbacks = []
for ewc_term_creator in ewc_term_creators:
    for k,v in ewc_term_creator.callback_dict.items():
        callbacks.append(v)

column_names = ["Epoch", "EWC Method", "Aggregation Level", "Threshold Value", "Loss", "Validation Loss"]
all_results_dataframe = pd.DataFrame(columns=column_names)
num_samples = 20
num_epochs = 25
sample_period = 5
sample_array = [(1/num_samples) * i for i in range(num_samples+1)]

def measure_val_loss_over_threshold(epoch_number, ewc_term_creators: List[EWC_Term_Creator]):
    epoch_results = pd.DataFrame(columns=column_names)
    for ewc_term_creator in ewc_term_creators:
        for aggregation_level in aggregation_levels:
            try:
                print(f"TRIAL {TRIAL_ID}: CURRENT TERM: {ewc_term_creator.ewc_method.name}, AGGREGATION LEVEL: {aggregation_level.name}")
                ewc_term = ewc_term_creator.create_term(ewc_lambda = 1, task=task)
                method_results = validation_loss_by_importance_threshold(task, ewc_term.omega_matrix, sample_array, aggregation_level=aggregation_level, show_plot=False)
                method_results["EWC Method"] = ewc_term_creator.ewc_method.name
                method_results["Aggregation Level"] = aggregation_level.name
                method_results["Epoch"] = epoch_number
                epoch_results = pd.concat([epoch_results, method_results], ignore_index=True)
            except Exception as e:
                print(f"TRIAL {TRIAL_ID}: EXCEPTION {e}")
                continue
    return epoch_results

epoch_index = 0
while epoch_index < num_epochs:
    epoch_index += sample_period
    task.train_on_task(epochs=sample_period, callbacks=callbacks)

    all_results_dataframe = pd.concat([all_results_dataframe, measure_val_loss_over_threshold(epoch_index, ewc_term_creators)], ignore_index=True)
    all_results_dataframe.to_csv(f"data/validation_loss_over_threshold_TRIAL_{TRIAL_ID}.csv")
    task.model.save(filepath=MODEL_SAVE_PATH)
