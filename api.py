from flask import Flask
from flasgger import Swagger, swag_from
from preprocessing import preprocess_sentence
from tensorflow import keras



app = Flask(__name__)
swagger = Swagger(app)

@app.route('/')
def hello():
    return "Welcome to Intent Classification"

@app.route('/api/')
@app.route('/api/docs/')
def show_docs():
    return "Documentation"

@app.route('/api/intent/')
def usage():
    return "Add a sentence to your request to get a prediction! \n Example: '/api/intent/Hello world!'"

@app.route('/api/intent/<sentence>/')
@swag_from('specs.yaml')
def predict(sentence):
    
    model = keras.models.load_model('./models/model_v1')
    x = preprocess_sentence(sentence)
    prediction = model.predict(x.reshape(1, 1, x.shape[0]))
    return str(prediction)
    

if __name__ == "__main__":
    
    app.run(debug=True)