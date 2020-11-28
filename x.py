# import flask.Flask
# from flask import Flask, url_for

import tensorflow as tf

from flask import Flask, render_template, request
app = Flask(__name__)


# tensorflow_version 2.x
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, MaxPooling1D, Dropout, LSTM, Bidirectional, SpatialDropout1D
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import utils
# from google.colab import files
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline 




num_words = 100
max_comment_len = 11

comments = pd.read_json( 'comments.json' )[0]

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(comments)
tokenizer.word_index
sequences = tokenizer.texts_to_sequences(comments)

# train = pad_sequences(sequences, maxlen=max_comment_len)
# x_train = train

model_lstm = Sequential()
model_lstm.add(Embedding(num_words, 128, input_length=max_comment_len))
model_lstm.add(SpatialDropout1D(0.5))
model_lstm.add(LSTM(40, return_sequences=True))
model_lstm.add(LSTM(40))
model_lstm.add(Dense(5, activation='sigmoid'))

model_lstm.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy', 'AUC'])
              
# model_lstm.summary()


model_lstm_save_path = 'best_model_lstm.h5'
# checkpoint_callback_lstm = ModelCheckpoint(model_lstm_save_path, 
#                                       monitor='val_accuracy',
#                                       save_best_only=True,
#                                       verbose=1)

# history_lstm = model_lstm.fit(x_train, 
#                               y_train, 
#                               epochs=2,
#                               batch_size=8,
#                               validation_split=0.1,
#                               callbacks=[checkpoint_callback_lstm])




model_lstm.load_weights(model_lstm_save_path)





def predict(text):
    
    sequence = tokenizer.texts_to_sequences([text])
    data = pad_sequences(sequence, maxlen=max_comment_len)
    result = model_lstm.predict(data)
    answ = ''
    answ = answ + str(result[0][0])[0:5] +' '
    answ = answ + str(result[0][1])[0:5] +' '
    answ = answ + str(result[0][2])[0:5] +' '
    answ = answ + str(result[0][3])[0:5] +' '
    answ = answ + str(result[0][4])[0:5]
    
    return answ



@app.route("/enbot", methods=['POST'])
def get_answer(name=None):
    text = request.json
    pr = predict(text)
    return pr
    # return "answer" + text



# @app.route("/enbot", methods=['GET'])
# def get_form(name=None):
#     return render_template('index.html', name=name)
    

if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=4567)
# app.run(host='0.0.0.0', port=4567)