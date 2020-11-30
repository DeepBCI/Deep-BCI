#CNN-LSTM CorNET
input1 = tf.keras.layers.Input(shape=data_generator.input_shape, name='input_1')
path1 = input1
#Conv-2
path1 = tf.keras.layers.Conv1D(80, 80, activation='relu')(path1)
path1 = tf.keras.layers.MaxPooling1D(4)(path1)
#Conv-2
path1 = tf.keras.layers.Conv1D(80, 80, activation='relu')(path1)
path1 = tf.keras.layers.MaxPooling1D(4)(path1)