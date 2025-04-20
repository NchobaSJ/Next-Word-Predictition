#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load the tokenizer and model
with open('friends1.txt', 'r', encoding='utf-8') as file:
    text = file.read()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

model = load_model('trained_language_model.h5')

def predict_next_words(seed_text, next_words=3):
    max_sequence_len = model.input_shape[1]
    
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=-1)[0]
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
                
        seed_text += " " + output_word
    
    return seed_text

# Test the function
if __name__ == "__main__":
    print("Next Word Prediction Test")
    print("Enter 'quit' to exit")
    
    while True:
        seed_text = input("\nEnter a seed phrase: ")
        if seed_text.lower() == 'quit':
            break
            
        try:
            num_words = int(input("How many words to predict? "))
            result = predict_next_words(seed_text, num_words)
            print("\nPredicted text:")
            print(result)
        except ValueError:
            print("Please enter a valid number for word count")