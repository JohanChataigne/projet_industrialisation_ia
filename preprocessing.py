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
        
        # Keep track of the removed sentences' intents during preprocessing
        self.intents_to_remove = []
    
    
    def clean(self):
        '''
        Remove some special characters and split sentences
        '''

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

                
    def vectorize(self):
        '''
        Transform words into their vector representation
        '''

        null_vector = np.zeros(300)

        for s in self.clean_text:
            vects = [w.vector if w.has_vector else null_vector for w in s]
            self.vectorized_text.append(vects)

    
    def pad(self):
        '''
        Add padding to sentences to have same size datas
        '''
        self.padded_sentences = pad_sequences(self.vectorized_text, dtype='float32', padding='post')

    
    def preprocess_sentences(self):
        '''
        Apply the whole pipeline to input sentences and return them as numpy array object
        '''
        self.clean()
        self.vectorize()
        self.pad()
        self.preprocessed_sentences = np.asarray(self.padded_sentences)


    def intent2vec(self, intent):
        '''
        One hot encode one intent (take string representation of the label)
        '''
        assert intent in intents

        idx = intents.index(intent)
        vec = np.zeros(len(intents))
        vec[idx] = 1
        return vec
    
    
    def clean_intents(self):
        '''
        Remove intents of removed sentences
        '''
        # Reverse to remove the right elements (no error due to index shifting)
        for idx in reversed(self.intents_to_remove):
            self.intents.pop(idx)
        
    
    def preprocess_intents(self):
        '''
        Apply one hot encoding to all the intents
        '''
        self.clean_intents()
        self.preprocessed_intents = np.asarray(list(map(self.intent2vec, self.intents)))
    
    
    def save_dataset(self, path):
        '''
        Save the dataset as a dataframe in a pickle file
        '''
        
        # Regroup labels and datas in one structure
        data = []
        for i in range(self.preprocessed_sentences.shape[0]):
            data.append([self.preprocessed_sentences[i], self.preprocessed_intents[i]])
            
        data = np.array(data)
        
        self.df_dataset = pd.DataFrame(data, columns=['sentence', 'intent'])
        
        if path is not None:
            with open(path, 'wb') as f:
                pickle.dump(self.df_dataset, f)
        
    
    def preprocess(self, path=None):
        '''
        Apply whole preprocessing to dataset
        '''
        
        if path is not None and os.path.exists(path):
            with open(path, 'rb') as f:
                self.df_dataset = pickle.load(f)
        else:
            self.preprocess_sentences()
            self.preprocess_intents()
            self.save_dataset(path)