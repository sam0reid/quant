
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pathlib 
import tensorflow as tf
import tensorflow_probability as tfp

from sklearn.preprocessing import MinMaxScaler

def getFeaturesTargets(ticker: str, period: int):
    """
    load x, y examples from memory
    """
    sequence_path = 'sequences'
    try:
        x = np.load(pathlib.Path(sequence_path) / ticker / f"x_{period}.npy")
        y = np.load(pathlib.Path(sequence_path) / ticker / f"y_{period}.npy")
    except FileNotFoundError:
        print(f"No datasets for ticker {ticker}, and period {period}.")
        return None, None
    return x, y

def getCompiledModel():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.GRU(units=512,
                                  return_sequences=True,
                                  input_shape=(None, 1,)))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    optimizer = tf.keras.optimizers.RMSprop(lr=1e-3)
    model.compile(loss=tf.keras.losses.mean_squared_error, 
                  optimizer=optimizer)
    print(model.summary())
    return model

def train():
    ticker = 'DOV'
    train_split = 0.9

    x, y = getFeaturesTargets(ticker, 10)
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    
    num_data = x.shape[0]
    num_train = int(train_split*num_data)

    x_train = x[0:num_train]
    y_train = y[0:num_train]

    x_test = x[num_train:]
    y_test = y[num_train:]

    print(f"Training x shape {x_train.shape}")
    print(f"Training y shape {y_train.shape}")
    print(f"Test x shape {x_test.shape}")
    print(f"Test y shape {y_test.shape}")

    x_scaler = MinMaxScaler()
    x_train_scaled = x_scaler.fit_transform(x_train)
    x_test_scaled = x_scaler.transform(x_test)

    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    def batchGenerator(batch_size: int, sequence_length: int):
        """
        Generator function for creating random batches of training data.
        """

        while True:
            x_shape = (batch_size, sequence_length, 1)
            x_batch = np.zeros(shape=x_shape, dtype=np.float16)

            y_shape = (batch_size, sequence_length, 1)
            y_batch = np.zeros(shape=y_shape, dtype=np.float16)

            for i in range(batch_size):
                idx = np.random.randint(num_train - sequence_length)
                x_batch[i] = x_train_scaled[idx:idx+sequence_length]
                y_batch[i] = y_train_scaled[idx:idx+sequence_length]
            
            yield (x_batch, y_batch)

    batch_size = 20
    sequence_length = 60 * 8 * 2

    generator = batchGenerator(batch_size=batch_size,
                               sequence_length=sequence_length)

    x_batch, y_batch = next(generator)
    print(f"x batch shape {x_batch.shape}")
    print(f"y batch shape {y_batch.shape}")

    validation_data = (np.expand_dims(x_test_scaled, axis=0),
                       np.expand_dims(y_test_scaled, axis=0))


    path_checkpoint = 'checkpoint.keras'
    callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint,
                                                            monitor='val_loss',
                                                            verbose=1,
                                                            save_weights_only=True,
                                                            save_best_only=True)
    callback_tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs/',
                                                    histogram_freq=0,
                                                    write_graph=False)

    model = getCompiledModel()
    model.fit(x=generator,
             epochs=100,
             steps_per_epoch=100,
             validation_data=validation_data,
             callbacks=[callback_checkpoint,
                        callback_tensorboard])

if __name__ == "__main__":
    train()
    