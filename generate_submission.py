import pathlib as path

import numpy as np
import pandas as pd
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers


class TestDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, image_list, batch_size=128, dim=(200, 200), n_channels=1):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.image_list = image_list
        self.n_channels = n_channels

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.image_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        # Find list of IDs
        temp_image_list = self.image_list[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X = self.__data_generation(temp_image_list)

        return X


    def __data_generation(self, temp_image_list):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((len(temp_image_list), *self.dim, self.n_channels))

        # Generate data
        for i, j in enumerate(temp_image_list):
            # Store sample
            img = np.array(Image.open(j))[200:400, 150:350]
            X[i,] = np.expand_dims(img, axis= 2)

        return X

def sigmoid(X):
    return 1/(1+np.exp(-X))

def get_class(X):
    return [1 if x < 0.5 else 0 for x in X]

def quantize_energy(energy):
    energy_levels = [1, 3, 6, 10, 20, 30]
    abs_error = [abs(energy_level - energy) for energy_level in energy_levels]
    quantized_energy = energy_levels[np.argmin(abs_error)]

    return quantized_energy

public_test_folder = path.Path('./tests/public_test/')
private_test_folder = path.Path('./tests/private_test/')

public_test_images = list(public_test_folder.glob('*.png'))
private_test_images = list(private_test_folder.glob('*.png'))

test_images = public_test_images + private_test_images

model = keras.Sequential(
    [
        layers.Conv2D(16, (5,5), strides=(1,1), padding='same', activation='relu', input_shape=(200, 200, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(10, 10), strides= 5),
        layers.Conv2D(32, (3,3), strides=(1,1), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(10, 10), strides= 5),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(2, activation=None),
    ]
)

model_weights_path = './track2_weights.86-77.06.hdf5'

model.load_weights(model_weights_path)


print('model loaded')
# Parameters
params = {'dim': (200, 200),
          'batch_size': 4,
          'n_channels': 1}

# Generators
test_generator = TestDataGenerator(test_images, **params)

preds = model.predict(test_generator)

class_preds = get_class(sigmoid(preds[:,0]))
energy_preds = preds[:,1]

test_df = pd.DataFrame(zip(test_images, class_preds, energy_preds), columns= ['image_paths', 'classification_predictions', 'regression_predictions'])

test_df['regression_predictions'] = test_df['regression_predictions'].apply(quantize_energy)

test_df['id'] = test_df['image_paths'].apply(lambda x: str(x).split('/')[-1].split('.')[0])

test_df[['id', 'classification_predictions', 'regression_predictions']].to_csv('submission.csv', index=False)