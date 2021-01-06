from flask import Flask
from preprocessing import preprocess_sentence



app = Flask(__name__)

@app.route('/')
def hello():
    return "Welcome to Intent Classification"

@app.route('/api/')
@app.route('/api/docs/')
def show_docs():
    return "Documentation"

@app.route('/api/intent/<sentence>/')
def predict(sentence):
    
    return str(preprocess_sentence(sentence))
    

if __name__ == "__main__":
    
    app.run(debug=True)