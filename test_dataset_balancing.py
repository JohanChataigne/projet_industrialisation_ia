import dataset_balancing as db
import pandas as pd

balance = db.Balance(pd.read_json('datas/training_set.json'))

def test_oversample():
    
    balance.oversample()
        
    assert len(balance.sentences) == 17852
    assert len([x for x in balance.intents if x == "irrelevant"]) == 3852
    assert len([x for x in balance.intents if x != "irrelevant"]) == 14000
    assert len([x for x in balance.intents if x == "find-restaurant"]) == 2000
    
def test_undersample():
    
    balance.undersample()
    
    assert len(balance.sentences) == 16000
    assert len([x for x in balance.intents if x == "irrelevant"]) == 2000
    