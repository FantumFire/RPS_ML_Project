import numpy as np
import keras
from keras import layers, optimizers, losses


def model_setup():
    
    encoder_vocab = 3
    decoder_vocab = 3

    encoder_input = layers.Input(shape=(None,))
    encoder_embedded = layers.Embedding(input_dim=encoder_vocab, output_dim=64)(encoder_input)

    output, state_h, state_c = layers.LSTM(64, return_state=True, name="encoder")(encoder_embedded)
    encoder_state = [state_h, state_c]

    decoder_input = layers.Input(shape=(None,))
    decoder_embedded = layers.Embedding(input_dim=decoder_vocab, output_dim=64)(decoder_input)

    decoder_output = layers.LSTM(64, name="decoder")(decoder_embedded, initial_state=encoder_state)
    output = layers.Dense(3, activation="softmax", name="output")(decoder_output)

    model = keras.Model([encoder_input, decoder_input], output)
    
    return model

model = model_setup()

model.save('model_name.keras')

'''

# Rock or R = 0
# Paper or P = 1
# Scissors or S = 2

encoder_data = np.array([[0, 0, 1, 2, 0, 1, 1, 2, 2], [1, 2, 2, 0, 1, 0, 2, 2, 1], [2, 0, 0, 2, 1, 2, 0, 1, 0], [0, 1, 2, 1, 1, 0, 2, 2, 0]])
decoder_data = np.array([[1], [2], [0], [1]])
target_data = np.array([[2], [0], [1], [2]])


model.compile(optimizer=optimizers.Adam(learning_rate=0.005), loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
model.fit([encoder_data, decoder_data], target_data, epochs=2, batch_size=1)

model_label = input("Insert name for model: ")

model.save(f'{model_label}.keras')
'''