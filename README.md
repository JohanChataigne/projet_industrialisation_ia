# Software engineering for AI project : Intent classification

Authors: [Chataigner Johan](https://github.com/JohanChataigne), [Germon Paul](https://github.com/pgermon), and [Martin Hugo](https://github.com/ScarfZapdos).

This project is based on its minimal version which can be found [here](https://hub.docker.com/r/wiidiiremi/projet_industrialisation_ia_3a).

The application offers to the users a classification model that takes their demands (french sentences) as input and classifies them into intents.


## Install and run the project

0. Install the requirements needed: `pip install -r requirements.txt`
1. Commands to run the application locally: 
   - `export FLASK_APP=app/app.py`
   - `python3 -m flask run` (debug server)
   - `python3 app/app.py` (production server with waitress)

2. Deploy application in a docker image:  
   - `docker build -t <img_name> <directory>`
   - `docker run -p 8080:8080 <img_name>`
    
3. Get the docker image on DockerHub and run it:
   - `docker pull ...`
   - `docker run -p 8080:8080 ...`



## Content of the project

### Repository tree organization

The repository is composed of the following files and folders:

📦projet_industrialisation_ia  
 ┣ 📂app : contains the files to run the application  
 ┣ 📂datas : contains the json datasets used to train the model  
 ┣ 📂models : contains the model saved  
 ┣ 📂notebooks  
 ┃ ┣ 📜model_v1.ipynb : notebook that trains the model  
 ┃ ┗ 📜project_analysis.ipynb : notebook in which we analyze the datasets and model provided  
 ┣ 📂obsolete_notebooks  
 ┣ 📂preprocessed_data : used to contain files storing the preprocessed data  
 ┣ 📂preprocessing  
 ┃ ┣ 📜dataset_balancing.py : contains functions used to balance the dataset   
 ┃ ┗ 📜preprocessing.py : contains functions used to preprocessed the dataset  
 ┣ 📂test : contains unit tests files   
 ┣ 📂threshold :  
   ┣ 📜threshold.py : contains functions used to compute the best threshold to be applied for our application    
 ┃ ┗ 📜best_threshold : store the value of the best   threshold computed  
 ┣ 📜Dockerfile  
 ┣ 📜README.md  
 ┗ 📜requirements.txt  

### Analysis and visualizations

The first step of the project is to analyze the minimal version provided to find out what it misses and what can be improved.  
In the notebook `project_analysis.ipynb` we make many visualizations and analysis on the given datasets and model in order to identify the most important metrics to be used to evaluate the performances of the model.  

### Data preprocessing

Seeing the analysis we have made on the datasets provided, we chose to balance the dataset in order to train our model. To do so we realised an oversampling of the less represented intents and an undersampling of the intent *irrelevant* in order to have 2000 samples of each intent. 

Then, in order to train our model as well as well as possible we had to preprocess the sentences of the dataset. We chose to use the SpaCy library to help us in this task. The preprocessing principally consists in:
- **removing special characters and determiners**: characters like emojis or punctuation signs and determiners like "a" are not that much usefull to identify the intent of a sentence. However we kept the accents which are widely used in French and the "€" symbol because it can help the model to identify the *purchase* intent for example.
- **vectorizing sentences**: we firstly used word vectors provided by SpaCy but the length of the sentences allowed was limited because the inputs of a model must always have the same shape. Therefore we chose to use sentence vectors which correspond to the mean of the vector of each word in the sentence.
- **one-hot encoding the intents**
  
We chose not to use the lemmatizer provided by SpaCy because we were receiving strange results for some words, maybe because the SpaCy french NLP model is less performant than the english one.

### Model training

After preprocessing the data, we tried to build a performant model in order to classify french sentences in a set of 8 different intents. Like the majority of text analysis Deep Learning models, our model is composed of a recurrent layer (Bidirectionnal LSTM) followed by several fully connected layers for classification. The output layer uses a *softmax* activation function in order to generate a set of probabilities matching the 8 intents. The model is trained using the *categorical crossentropy* loss function as the problem is a multi-class classification problem.

### Model evaluation

After training our model we had to evaluate its performance. We obtained the following performance metrics on testing set:

- Loss = 1.17
- Precision = 0.81
- Recall (weighted avg) = 0.81
- F1-score = 0.82
- F0.5-score = 0.83



## Future improvements