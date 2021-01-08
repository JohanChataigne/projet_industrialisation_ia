import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import spacy
import fr_core_news_md
import pickle
import typing as t


#pip install https://github.com/explosion/spacy-models/releases/download/fr_core_news_md-2.0.0/fr_core_news_md-2.0.0.tar.gz

# List of all intents in the same order as the model's output
intents = ["find-train", "irrelevant", "find-flight", "find-restaurant", "purchase", "find-around-me", "provide-showtimes", "find-hotel"]
    
class Preprocessor:
    
    def __init__(self, dataset: t.Iterable) -> t.NoReturn:
        
        #os.system('python -m spacy download fr_core_news_md')
        self.nlp = fr_core_news_md.load()
        #self.nlp = spacy.load('fr_core_news_md')
        
        self.sentences = list(dataset['sentence'])
        self.intents = list(dataset['intent'])
        self.clean_text = []
        self.vectorized_text = []
        
        # Keep track of the removed sentences' intents during preprocessing
        self.intents_to_remove = []
    
    
    def clean(self) -> t.NoReturn:
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

                
    def vectorize(self) -> t.NoReturn:
        '''
        Transform words into their vector representation
        '''

        null_vector = np.zeros(300)
        
        for s in self.clean_text:
            v = s.vector if s.has_vector else null_vector
            self.vectorized_text.append(v)
            
        self.vectorized_text = np.asarray(self.vectorized_text)
        self.preprocessed_sentences = self.vectorized_text

    
    def preprocess_sentences(self) -> t.NoReturn:
        '''
        Apply the whole pipeline to input sentences and return them as numpy array object
        '''
        self.clean()
        self.vectorize()


    def intent2vec(self, intent: t.Iterable) -> t.Iterable:
        '''
        One hot encode one intent (take string representation of the label)
        '''
        assert intent in intents

        idx = intents.index(intent)
        vec = np.zeros(len(intents))
        vec[idx] = 1
        return vec
    
    
    def clean_intents(self) -> t.NoReturn:
        '''
        Remove intents of removed sentences
        '''
        # Reverse to remove the right elements (no error due to index shifting)
        for idx in reversed(self.intents_to_remove):
            self.intents.pop(idx)
        
    
    def preprocess_intents(self) -> t.NoReturn:
        '''
        Apply one hot encoding to all the intents
        '''
        self.clean_intents()
        self.preprocessed_intents = np.asarray(list(map(self.intent2vec, self.intents)))
    
    
    def save_dataset(self, path: str) -> t.NoReturn:
        '''
        Save the dataset as a dataframe in a pickle file
        '''
        
        # Regroup labels and datas in one structure
        data = []
        #print(self.preprocessed_sentences.shape)
        #print(self.preprocessed_intents.shape)
        assert len(self.preprocessed_sentences) == len(self.preprocessed_intents)
        
        for i in range(len(self.preprocessed_sentences)):
            data.append([self.preprocessed_sentences[i], self.preprocessed_intents[i]])
            
        data = np.array(data)
        
        self.df_dataset = pd.DataFrame(data, columns=['sentence', 'intent'])
        
        if path is not None:
            with open(path, 'wb') as f:
                pickle.dump(self.df_dataset, f)
        
    
    def preprocess(self, path: str=None, force: bool=False) -> t.NoReturn:
        '''
        Apply whole preprocessing to dataset
        '''
        
        if path is not None and os.path.exists(path) and not force:
            with open(path, 'rb') as f:
                self.df_dataset = pickle.load(f)
        else:
            self.preprocess_sentences()
            self.preprocess_intents()
            self.save_dataset(path)
            
            
def preprocess_sentence(sentence: str) -> t.Iterable:
    
    nlp = fr_core_news_md.load()
    
    # remove special characters
    clean_sentence = re.sub(r'[^ A-Za-z0-9éèàêî€]', '', sentence)
    doc_s = nlp(clean_sentence)

    # remove determiners (i.e. pos = 90 or pos_ = 'DET')
    remaining_words = list(filter(lambda x: x.pos != 90, doc_s))
    # build new doc object
    str_s = " ".join(list(map(lambda x: x.text, remaining_words)))
    final_sentence = nlp(str_s)

    null_vector = np.zeros(300)

    vect = np.asarray(final_sentence.vector if final_sentence.has_vector else null_vector)

    return vect

    
