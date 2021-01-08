import sys
import os
import pickle
sys.path.append('./preprocessing')

from flask import Flask
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
    
    model = keras.models.load_model('./models/model_v1')
    x = preprocess_sentence(sentence)
    prediction = model.predict(x.reshape(1, 1, x.shape[0]))
    intent = get_predicted_intent(prediction, threshold)
    return intent_pretty_print(intent)
    

if __name__ == "__main__":
    
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)