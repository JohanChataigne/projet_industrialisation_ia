import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import spacy
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# List of all intents in the same order as the model's output
intents = ["find-train", "irrelevant", "find-flight", "find-restaurant", "purchase", "find-around-me", "provide-showtimes", "find-hotel"]
    
class Preprocessor:
    
    def __init__(self, dataset):
        
        os.system('python -m spacy download fr_core_news_md')
        self.nlp = spacy.load('fr_core_news_md')
        
        self.sentences = list(dataset['sentence'])
        self.intents = list(dataset['intent'])
        self.clean_text = []
        self.vectorized_text = []
        self.padded_sentences = []
        
        # Keep track of the intents of the removed sentences during preprocessing
        self.intents_to_remove = []
    
    # Remove some special characters and split sentences
    def clean(self):

        for i, s in enumerate(self.sentences):
            # remove special characters
            clean_sentence = re.sub(r'[^ A-Za-z0-9éèàêî€]', '', s)
            doc_s = self.nlp(clean_sentence)
            
            # remove determiners (i.e. pos = 90 or pos_ = 'DET')
            remaining_words = list(filter(lambda x: x.pos != 90, doc_s))
            # build new doc object
            str_s = " ".join(list(map(lambda x: x.text, remaining_words)))
            final_sentence = self.nlp(str_s)

            # Exclude empty sentences
            if len(final_sentence) > 0: self.clean_text.append(final_sentence)
            else: self.intents_to_remove.append(i)

    # Transform words into their vector representation
    def vectorize(self):

        null_vector = np.zeros(300)

        for s in self.clean_text:
            vects = [w.vector if w.has_vector else null_vector for w in s]
            self.vectorized_text.append(vects)

    # Add padding to sentences to have same size datas
    def pad(self):
        self.padded_sentences = pad_sequences(self.vectorized_text, dtype='float32', padding='post')

    # Apply the whole pipeline to input sentences and return them as numpy array object
    def preprocess_sentences(self):
        self.clean()
        self.vectorize()
        self.pad()
        self.preprocessed_sentences = np.asarray(self.padded_sentences)

    # One hot encode one intent (take string representation of the label)
    def intent2vec(self, intent):
        assert intent in intents

        idx = intents.index(intent)
        vec = np.zeros(len(intents))
        vec[idx] = 1
        return vec
    
    # Remove intents of removed sentences
    def clean_intents(self):
        # Reverse to remove the right elements (no error due to index shifting)
        for idx in reversed(self.intents_to_remove):
            self.intents.pop(idx)
        
    # Apply one hot encoding to all the intents
    def preprocess_intents(self):
        self.clean_intents()
        self.preprocessed_intents = np.asarray(list(map(self.intent2vec, self.intents)))
    
    # Save the dataset as a dataframe in a pickle file
    def save_dataset(self):
        self.df_dataset = pd.DataFrame(self.preprocesses_sentences, self.preprocessed_intents,
                                       columns=['sentence', 'intent'])
        with open('pickle_preprocessed_dataset', 'wb') as f:
            pickle.dumb(self.df_dataset, f)
        
    # Apply whole preprocessing to dataset
    def preprocess(self):
        self.preprocess_sentences()
        self.preprocess_intents()
        
        # TODO build preprocessed dataset object
        self.save_dataset()

            
        
        
    
