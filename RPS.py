import numpy as np
import keras
from keras import layers, optimizers, losses

def player(prev_play, opponent_history=["R"]):
    move_to_int = {"R": 0, "P": 1, "S": 2}
    int_to_move = {0: "R", 1: "P", 2: "S"}
    
    if prev_play == "":
        prev_play = "R"
    opponent_history.append(prev_play)

    
    int_history = [move_to_int[move] for move in opponent_history]

    encoder_input = np.array([int_history[-9:]]) 
    decoder_input = np.array([[move_to_int[prev_play]]]) 

    def load_model():
        
        model = keras.models.load_model('insert_model_name_here.keras') # Choose your model here
        
        return model
    
    def default_model():
    
        encoder = 3
        decoder = 3

        encoder_input = layers.Input(shape=(None,))
        encoder_embedded = layers.Embedding(input_dim=encoder, output_dim=64)(encoder_input)

        output, state_h, state_c = layers.LSTM(64, return_state=True, name="encoder")(encoder_embedded)
        encoder_state = [state_h, state_c]

        decoder_input = layers.Input(shape=(None,))
        decoder_embedded = layers.Embedding(input_dim=decoder, output_dim=64)(decoder_input)

        decoder_output = layers.LSTM(64, name="decoder")(decoder_embedded, initial_state=encoder_state)
        output = layers.Dense(3, activation="softmax", name="output")(decoder_output)

        model = keras.Model([encoder_input, decoder_input], output)
        
        return model
    
    loaded_model = load_model()
    model = default_model()

    decoder_output = model.predict([encoder_input, decoder_input]) # change the model between (loaded_model) or (model) based on who the player is going against
    predicted_opp_move = np.argmax(decoder_output)
    predicted_opp_move = predicted_opp_move % 3

    def counter_move(opponent_move):
        if opponent_move == move_to_int["R"]:  
            return move_to_int["P"]
        elif opponent_move == move_to_int["P"]:
            return move_to_int["S"]
        elif opponent_move == move_to_int["S"]:
            return move_to_int["R"]
    
    counter_int = counter_move(predicted_opp_move)
    counter_str = int_to_move[counter_int]
    
    return counter_str
