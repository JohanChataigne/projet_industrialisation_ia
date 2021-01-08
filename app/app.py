import sys
import os
import pickle
sys.path.append('./preprocessing')

from flask import Flask, jsonify
from flasgger import Swagger, swag_from

from preprocessing import preprocess_sentence
from tensorflow import keras
from utils import *
import markdown
import markdown.extensions.fenced_code
    
        
# Load threshold if exists
if os.path.exists('./threshold/best_threshold'):
    with open('./threshold/best_threshold', 'rb') as f:
        threshold = pickle.load(f)
else:
    threshold = None
    
    
# Init flask app
app = Flask(__name__)
app.config.from_object('config.Config')
swagger = Swagger(app)

# Load trained model
model = keras.models.load_model('./models/model_v1')

# API routes
@app.route('/')
@app.route('/api/')
def home():
    """Home route"""
    
    md_file = open('./app/home.md', 'r')
    md_template_string = markdown.markdown(
          md_file.read(), extensions=["fenced_code"]
    )

    return md_template_string

@app.route('/apidocs/')
def apidocs():
    """Documentation route"""
    pass

@app.route('/api/intent/<sentence>/')
@swag_from('intent.yaml')
def predict(sentence):
    
    response_dict = {}
    
    # Preprocess given sentence
    x = preprocess_sentence(sentence)
    
    # Get predicted probabilities
    prediction = model.predict(x.reshape(1, 1, x.shape[0])).flatten()
    
    # Compute predicted intent
    intent = get_predicted_intent(prediction, threshold)
    
    response_dict['prediction'] = build_prediction_dict(prediction)
    response_dict['intent'] = intent
    response_dict['message'] = intent_pretty_print(intent)
    
    return jsonify(response_dict)
    

if __name__ == "__main__":
    
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)