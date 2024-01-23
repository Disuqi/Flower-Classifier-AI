import os
import re
import pandas as pd
from enum import Enum
from keras.preprocessing.image import ImageDataGenerator

class FlowerType(Enum):
    Daffodil = 0
    Sunflower = 9
    Danedelion = 12

# Edit these constants to change the data used or place the data in the expected folders.
IMAGES_PER_FLOWER_TYPE = 80
BASE_DIR = "./data/17flowers"
FILES_TXT = "files.txt"

def get_flower_type(filename):
    match = re.search(r'\d+', filename)
    if match:
        image_index = int(match.group()) - 1
    else:
        raise ValueError("Invalid filename: No number found.")
    
    flower_type_value = (image_index // IMAGES_PER_FLOWER_TYPE)
    return FlowerType(flower_type_value)

def get_flowers_dataframes(train_percentage = 0.8, test_percentage = 0):
    """
    Splits the flower data into training, testing, and validation sets.

    Args:
        train_percentage (float, optional): The proportion of the data to use for training. Defaults to 0.8.
        test_percentage (float, optional): The proportion of the data to use for testing. Defaults to 0.1.

    Returns:
        tuple: A tuple containing three pandas DataFrames. The first DataFrame contains the training data, 
               the second DataFrame contains the testing data, and the third DataFrame contains the validation data.

    Raises:
        ValueError: If train_percentage + test_percentage is greater than 1.
    """
    data = {
        FlowerType.Daffodil: pd.DataFrame(columns=['filename', 'class']),
        FlowerType.Sunflower: pd.DataFrame(columns=['filename', 'class']),
        FlowerType.Danedelion:  pd.DataFrame(columns=['filename', 'class'])
    }

    if(train_percentage + test_percentage > 1):
        raise ValueError("Train and test percentage must not be greater than 1")
    if(test_percentage == 0):
        test_percentage = (1 - train_percentage)/2 # default is 0.1

    with open(os.path.join(BASE_DIR, FILES_TXT)) as file:
        lines = file.readlines()
        for flower in FlowerType:
            starting_index = flower.value * IMAGES_PER_FLOWER_TYPE
            for i in range(starting_index, starting_index + IMAGES_PER_FLOWER_TYPE):
                line = lines[i].strip()
                data[flower] = pd.concat([data[flower], pd.DataFrame({'filename': [line], 'class': [flower.name]})], ignore_index=True)

    train_data = pd.DataFrame(columns=['filename', 'class'])
    test_data = pd.DataFrame(columns=['filename', 'class'])
    val_data = pd.DataFrame(columns=['filename', 'class'])

    for flower in FlowerType:
        flower_data = data[flower]
        flower_data = flower_data.sample(frac=1).reset_index(drop=True)

        data_len = len(flower_data)
        train_len = int(data_len * train_percentage)
        test_len = int(data_len * test_percentage)

        train_data = pd.concat([train_data, flower_data[:train_len]], ignore_index=True)
        test_data = pd.concat([test_data, flower_data[train_len:train_len + test_len]], ignore_index=True)
        val_data = pd.concat([val_data, flower_data[train_len + test_len:]], ignore_index=True)

    train_data = train_data.sample(frac=1).reset_index(drop=True)
    test_data = test_data.sample(frac=1).reset_index(drop=True)
    val_data = val_data.sample(frac=1).reset_index(drop=True)
    return train_data, test_data, val_data

def get_image_generators_from_dataframes(train_data, test_data, val_data, target_size=(250, 250)):
    """
    Creates image generators from pandas DataFrames.

    Args:
        train_data (pandas.DataFrame): DataFrame containing the training data.
        test_data (pandas.DataFrame): DataFrame containing the testing data.
        val_data (pandas.DataFrame): DataFrame containing the validation data.

    Returns:
        tuple: A tuple containing three ImageDataGenerators. The first generator is for training data, 
               the second generator is for testing data, and the third generator is for validation data.

    Raises:
        ValueError: If train_data, test_data, or val_data is not a pandas DataFrame, or if any of the DataFrames do not have exactly two columns.
    """
    if not isinstance(train_data, pd.DataFrame) or not isinstance(test_data, pd.DataFrame) or not isinstance(val_data, pd.DataFrame):
        raise ValueError("train_data, test_data, and val_data must be pandas DataFrames.")
    
    if train_data.shape[1] != 2 or test_data.shape[1] != 2 or val_data.shape[1] != 2:
        raise ValueError("train_data, test_data, and val_data must have exactly two columns.")

    train_datagen = ImageDataGenerator(rescale=1./255,
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255, 
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255,
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        horizontal_flip=True)

    train_generator = train_datagen.flow_from_dataframe(train_data, BASE_DIR, target_size=target_size, batch_size=10, class_mode='categorical')
    test_generator = test_datagen.flow_from_dataframe(test_data, BASE_DIR, target_size=target_size, batch_size=10, class_mode='categorical')
    val_generator = val_datagen.flow_from_dataframe(val_data, BASE_DIR, target_size=target_size, batch_size=10, class_mode='categorical', shuffle=False)

    return train_generator, test_generator, val_generator


train, test, val = get_flowers_dataframes()
train_gen, test_gen, val_gen = get_image_generators_from_dataframes(train, test, val)