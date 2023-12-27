import os;
from keras import models;

MODELS_DIR = "./saved_models"

def save_model(model, name, overwrite = False):
    """
    Saves a model to the models directory.

    Args:
        model (keras.models.Model): The model to save.
        name (str): The name of the model.
        overwrite (bool, optional): Whether to overwrite the model if it exists. Defaults to False.

    Raises:
        ValueError: If `model` or `name` is None, or if `overwrite` is False and a model with the same name exists.
    """
    if model is None:
        raise ValueError("Model cannot be None.")
    if name is None:
        raise ValueError("Name cannot be None.")

    if not overwrite:
        existing_models = os.listdir(MODELS_DIR)
        if name in existing_models:
            raise ValueError("Model already exists. Set overwrite to True to overwrite the model.")

    model.save(os.path.join(MODELS_DIR, name))

def load_model(name):
    """
    Loads a model from the models directory.

    Args:
        name (str): The name of the model.

    Raises:
        ValueError: If `name` is None, or if no model with the given name exists.

    Returns:
        keras.models.Model: The loaded model.
    """
    if name is None:
        raise ValueError("Name cannot be None.")

    existing_models = os.listdir(MODELS_DIR)
    if name not in existing_models:
        raise ValueError("Model does not exist.")

    return models.load_model(os.path.join(MODELS_DIR, name))