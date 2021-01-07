from flask import Flask
from flasgger import Swagger, swag_from
from preprocessing import preprocess_sentence
from tensorflow import keras
from utils import *
import markdown
import markdown.extensions.fenced_code
    
import os
import pickle


# Load threshold if exists
if os.path.exists('./best_threshold'):
    with open('./best_threshold', 'rb') as f:
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
    
    md_file = open('./home.md', 'r')
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
    
    app.run(debug=True, port=8080)