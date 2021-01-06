from tensorflow import keras
import pickle
import numpy as np
from sklearn.metrics import fbeta_score

model = keras.models.load_model('./models/model_v1')

with open('./preprocessed_data/test_nlp_preprocessed', 'rb') as fp:
    test_set = pickle.load(fp)
    
x_test = np.array(list(test_set['sentence']))
y_test = np.array(list(test_set['intent']))

def apply_threshold(x, threshold):
    
    idx_max = np.argmax(x)
    return idx_max if x[idx_max] >= threshold else 1
    

def evaluate(threshold):
    
    preds = model.predict(x_test)
    
    y_pred = list(map(lambda x: apply_threshold(x, threshold), preds))
    
    y_test_true = list(map(lambda x: np.argmax(x), y_test))
        
    return fbeta_score(y_test_true, y_pred, beta=0.5, average='weighted')
    
    
def compute_threshold():
    
    scores = []
    thresholds = np.linspace(0, 1, 100)
    
    for t in thresholds:
        
        scores.append(evaluate(t))
        
    maximum = max(scores)
    
    return scores, maximum, thresholds[scores.index(maximum)]
        
        
    
    
scores, maximum, t = compute_threshold()

print(scores)
print(maximum)
print(t)
