import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pickle

class Balance:
    
    def __init__(self, dataset):
        
        self.sentences = dataset['sentence']
        self.intents = dataset['intent']
    
    
    # Oversample all the classes except irrelevant to 2000 samples each by duplicating samples
    def oversample(self):
        
        labels = np.unique(self.intents)
        oversampling_dict = {}
        for l in labels:
            if l == 'irrelevant': continue
            oversampling_dict[l] = 2000
            
        oversampler = RandomOverSampler(sampling_strategy=oversampling_dict)
        self.sentences, self.intents = oversampler.fit_resample(self.sentences, self.intents)
        
    # Undersample the irrelevant class to 2000 samples by removing samples
    def undersample(self):
        undersampler = RandomUnderSampler(sampling_strategy='majority')
        self.sentences, self.intents = undersampler.fit_resample(self.sentences, self.intents)
        
    # Save the dataset as a dataframe in a pickle file
    def save_dataset(self, path):
        self.df_dataset = pd.DataFrame(np.array([self.sentences, self.intents], dtype=object),
                                       columns=['sentence', 'intent'])
        
        if path is not None:
            with open(path, 'wb') as f:
                pickle.dumb(self.df_dataset, f)
        
    # Apply the oversampling and undersampling to the dataset
    def process_balance(self, path=None):
        
        if path is not None and os.path.exists(path):
            with open(path, 'rb') as f:
                self.df_dataset = pickle.load(f)
        else: 
            self.oversample()
            self.undersample()
            self.save_dataset(path)