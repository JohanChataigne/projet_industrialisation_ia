import utils as u
import numpy as np

def test_get_predicted_intent():
    
    probs = np.array([0, 0.1, 0.85, 0, 0, 0, 0.05, 0])
    
    assert u.get_predicted_intent(probs) == "find-flight"
    assert u.get_predicted_intent(probs, threshold=0.80) == "find-flight"
    assert u.get_predicted_intent(probs, threshold=0.90) == "irrelevant"
    
def test_intent_pretty_print():
     assert u.intent_pretty_print("irrelevant") == "I didn't understand what you want, your request is being sent to a human agent."