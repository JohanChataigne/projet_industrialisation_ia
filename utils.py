import numpy as np

# List of all intents in the same order as the model's output
intents = ["find-train", "irrelevant", "find-flight", "find-restaurant", "purchase", "find-around-me", "provide-showtimes", "find-hotel"]

def get_predicted_intent(probs, threshold=None):
    
    theorical_max = np.argmax(probs)
    
    probs = probs.flatten()
    
    if threshold is not None:
        intent = intents[theorical_max] if probs[theorical_max] >= threshold else "irrelevant"
    else:
        intent = intents[theorical_max]
        
    return intent

prints = {
        0: "I figured out that you want to find a train.",
        1: "I didn't understand what you want, your request is being sent to a human agent.",
        2: "I figured out that you want to find a flight.",
        3: "I figured out that you want to find a place to eat.",
        4: "I figured out that you want to buy something.",
        5: "I figured out that you want to find a place around you.",
        6: "I figured out that you want to get the schedule for a show.",
        7: "I figured out that you want to find a hotel."   
    }

def intent_pretty_print(intent):
    
    assert intent in intents
    
    return prints[intents.index(intent)]