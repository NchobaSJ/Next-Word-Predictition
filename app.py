#!/usr/bin/env python
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load model and tokenizer
model = None
tokenizer = None
total_words = 0
max_sequence_len = 0

def load_assets():
    global model, tokenizer, total_words, max_sequence_len
    
    # Load tokenizer
    with open('friends1.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1
    
    # Load model
    model = load_model('trained_language_model.h5')
    max_sequence_len = model.input_shape[1]

# Load assets when starting the app
load_assets()

def predict_next_words(seed_text, next_words=3):
    try:
        result = seed_text
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([result])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
            predicted_probs = model.predict(token_list, verbose=0)
            predicted_index = np.argmax(predicted_probs, axis=-1)[0]
            
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted_index:
                    output_word = word
                    break
                    
            result += " " + output_word
        
        return result
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return seed_text  # Return original text if prediction fails

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        seed_text = request.form.get('seed_text', '')
        num_words = request.form.get('num_words', '3')
        
        try:
            num_words = int(num_words)
            if num_words < 1:
                num_words = 1
            if num_words > 10:  # Limit to prevent long processing
                num_words = 10
        except ValueError:
            num_words = 3
            
        predicted_text = predict_next_words(seed_text, num_words)
        return render_template('index.html', 
                             seed_text=seed_text,
                             num_words=num_words,
                             predicted_text=predicted_text)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)