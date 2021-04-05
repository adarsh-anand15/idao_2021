import os

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers
from PIL import Image

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df, batch_size=32, dim=(200, 200), n_channels=1,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        df_temp = self.df.loc[indexes]

        # Generate data
        X, y = self.__data_generation(df_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.array(self.df.index)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, df_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, 2), dtype=float)

        # Generate data
        for i, j in enumerate(df_temp.index):
            # Store sample
            img = np.array(Image.open(df_temp.loc[j, 'image_path']))[200:400, 150:350]
            X[i,] = np.expand_dims(img, axis= 2) 

            # Store class
            y[i,0] = df_temp.loc[j, 'class']
            y[i,1] = df_temp.loc[j, 'energy']

        return X, y

def custom_loss(y_true, y_pred):
            
    # calculate loss, using y_pred
    y_true_class = y_true[:,0]
    y_pred_class = y_pred[:,0]

    y_pred_class = tf.clip_by_value(tf.math.sigmoid(y_pred_class), 1e-4, 1.0-1e-4)

    # -sum(y.log(y'))
    lossCobj = -tf.reduce_sum(tf.multiply(y_true_class, tf.math.log(y_pred_class)))
    
    # -sum((1-y).log(1-y'))
    lossCnoobj = -tf.reduce_sum(tf.multiply(tf.subtract(1.0, y_true_class), tf.math.log(tf.subtract(1.0, y_pred_class))))
        
    y_true_energy = y_true[:,1]
    y_pred_energy = y_pred[:,1]

    lossEnergy = tf.reduce_sum(tf.square(tf.subtract(y_true_energy, y_pred_energy)))

    total_loss = (lossCobj + lossCnoobj + lossEnergy)
        
    return total_loss

def custom_accuracy_metric(y_true, y_pred):
            
    # calculate loss, using y_pred
    y_true_class = tf.cast(y_true[:,0], tf.bool)
    y_pred_class = y_pred[:,0]

    y_pred_class = tf.clip_by_value(tf.math.sigmoid(y_pred_class), 1e-4, 1.0-1e-4)
    y_pred_class = tf.math.greater(y_pred_class, 0.5)

    return tf.reduce_sum(tf.cast(y_pred_class == y_true_class, tf.int32))/ len(y_true_class)

def custom_loss_metric(y_true, y_pred):

    # -sum(y.log(y'))
    # lossCobj = -tf.reduce_sum(tf.multiply(y_true_class, tf.math.log(y_pred_class)))
    
    # -sum((1-y).log(1-y'))
    # lossCnoobj = -tf.reduce_sum(tf.multiply(tf.subtract(1.0, y_true_class), tf.math.log(tf.subtract(1.0, y_pred_class))))
        
    y_true_energy = y_true[:,1]
    y_pred_energy = y_pred[:,1]

    lossEnergy = tf.reduce_sum(tf.abs(tf.subtract(y_true_energy, y_pred_energy)))

    # total_loss = (lossCobj + lossCnoobj + lossEnergy)
        
    return lossEnergy

er_folder = './idao_dataset/train/ER/'
er_images = os.listdir(er_folder)

nr_folder = './idao_dataset/train/NR/'
nr_images = os.listdir(nr_folder)

er_df = pd.DataFrame(er_images, columns=['image_name'])
er_df['image_path'] = er_folder + er_df['image_name']
er_df['class'] = 0
er_df['energy'] = er_df['image_name'].str.split('_').str[6].apply(lambda x: float(x))

nr_df = pd.DataFrame(nr_images, columns=['image_name'])
nr_df['image_path'] = nr_folder + nr_df['image_name']
nr_df['class'] = 1
nr_df['energy'] = nr_df['image_name'].str.split('_').str[7].apply(lambda x: float(x))

df = pd.concat([er_df, nr_df]).reset_index(drop=True)

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

opt = keras.optimizers.Adam(learning_rate=0.001, decay=1e-4)
model.compile(loss=custom_loss, optimizer=opt,metrics=['accuracy', custom_accuracy_metric, custom_loss_metric])

train_df, test_df = train_test_split(df, test_size=0.2, random_state=15)

# Parameters
params = {'dim': (200, 200),
          'batch_size': 128,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True}

# Generators
training_generator = DataGenerator(train_df, **params)
validation_generator = DataGenerator(test_df, **params)

checkpoint_filepath = './track2_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=False,
    save_freq='epoch')

model.fit(training_generator, epochs=100, validation_data=validation_generator, callbacks=[model_checkpoint_callback])


