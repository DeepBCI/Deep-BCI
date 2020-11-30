#CNN-LSTM CorNET
input1 = tf.keras.layers.Input(shape=data_generator.input_shape, name='input_1')
path1 = input1
#Conv-2
path1 = tf.keras.layers.Conv1D(32, 40, activation='relu')(path1)
path1 = tf.keras.layers.MaxPooling1D(4)(path1)
#Conv-2
path1 = tf.keras.layers.Conv1D(32, 40, activation='relu')(path1)
path1 = tf.keras.layers.MaxPooling1D(4)(path1)
#LSTM-1
path1 = tf.keras.layers.LSTM(32, return_sequences=True)(path1)
path1 = tf.keras.layers.Dropout(0.3)(path1)
#LSTM-2
path1 = tf.keras.layers.LSTM(32, return_sequences=True)(path1)
path1 = tf.keras.layers.Dropout(0.3)(path1)
#Normalization, Flatten
path1 = tf.keras.layers.BatchNormalization()(path1)
path1 = tf.keras.layers.Flatten()(path1)
#Dense layer
path1 = tf.keras.layers.Dense(32, activation='relu')(path1)
output1 = tf.keras.layers.Dense(data_generator.num_class, activation='softmax', name='output_1')(path1)