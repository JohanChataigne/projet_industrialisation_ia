import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pickle
import os
import typing as t

class Balance:
    
    def __init__(self, dataset) -> t.NoReturn:
        
        self.sentences = dataset['sentence'].to_numpy()
        self.intents = dataset['intent'].to_numpy()
    
    
    def oversample(self) -> t.NoReturn:
        '''
        Oversample all the classes except irrelevant to 2000 samples each by duplicating samples
        '''
        
        labels = np.unique(self.intents)
        oversampling_dict = {}
        for l in labels:
            if l == 'irrelevant': continue
            oversampling_dict[l] = 2000
            
        oversampler = RandomOverSampler(sampling_strategy=oversampling_dict)
        self.sentences, self.intents = oversampler.fit_resample(self.sentences.reshape(-1, 1), self.intents)
        self.sentences = self.sentences.flatten()
        
    
    def undersample(self) -> t.NoReturn:
        '''
        Undersample the irrelevant class to 2000 samples by removing samples
        '''
        undersampler = RandomUnderSampler(sampling_strategy='majority')
        self.sentences, self.intents = undersampler.fit_resample(self.sentences.reshape(-1, 1), self.intents)
        self.sentences = self.sentences.flatten()
        

    def save_dataset(self, path: str) -> t.NoReturn:
        '''
        Save the dataset as a dataframe in a pickle file
        '''
        
        # Regroup labels and datas in one structure
        data = []
        for i in range(len(self.sentences)):
            data.append([self.sentences[i], self.intents[i]])
            
        data = np.array(data)
        
        self.df_dataset = pd.DataFrame(data, columns=['sentence', 'intent'])
        
        if path is not None:
            with open(path, 'wb') as f:
                pickle.dump(self.df_dataset, f)
        
    
    def process_balance(self, path: str=None, force: bool=False) -> t.NoReturn:
        '''
        Apply the oversampling and undersampling to the dataset
        '''
        
        if path is not None and os.path.exists(path) and not force:
            with open(path, 'rb') as f:
                self.df_dataset = pickle.load(f)
        else: 
            self.oversample()
            self.undersample()
            self.save_dataset(path)