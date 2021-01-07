from tensorflow import keras
import pickle
import numpy as np
from sklearn.metrics import fbeta_score
import typing as t

model = keras.models.load_model('./models/model_v1')

with open('./preprocessed_data/test_nlp_preprocessed', 'rb') as fp:
    test_set = pickle.load(fp)
    
x_test = np.array(list(test_set['sentence']))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
y_test = np.array(list(test_set['intent']))


def apply_threshold(x, threshold: float) -> int:
    '''
    Return the index of the class predicted according to the given threshold
    '''
    
    idx_max = np.argmax(x)
    return idx_max if x[idx_max] >= threshold else 1
    

def evaluate(threshold: float) -> float:
    '''
    Evaluate the fbeta_score of the model according to the given threshold
    '''
    
    preds = model.predict(x_test)
    
    y_pred = list(map(lambda x: apply_threshold(x, threshold), preds))
    
    y_test_true = list(map(lambda x: np.argmax(x), y_test))
        
    return fbeta_score(y_test_true, y_pred, beta=0.5, average='weighted')
    
    
def compute_threshold() -> t.Tuple[t.List, float, float]:
    '''
    Compute the best threshold
    '''
    
    scores = []
    thresholds = np.linspace(0, 1, 100)
    
    for t in thresholds:
        
        scores.append(evaluate(t))
        
    maximum = max(scores)
    
    return scores, maximum, thresholds[scores.index(maximum)]
        
        
if __name__ == "__main__":

    scores, max_score, threshold = compute_threshold()
    
    print("scores :", scores)
    print("max_score =", max_score)
    print("best threshold =", threshold)
    
    with open('./best_threshold', 'wb') as f:
        pickle.dump(threshold, f)
