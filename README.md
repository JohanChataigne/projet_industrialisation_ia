# Software engineering for AI project : Intent classification

Authors: [Chataigner Johan](https://github.com/JohanChataigne), [Germon Paul](https://github.com/pgermon), and [Martin Hugo](https://github.com/ScarfZapdos).

This project is based on its minimal version which can be found [here](https://hub.docker.com/r/wiidiiremi/projet_industrialisation_ia_3a).

The application offers to the users a classification model that takes their demands (sentences) as input and classifies them into intents.


## Installation

Link to the docker image on DockerHub:


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

The first step of the project is to analyze the minimal version given to find out what it misses and what can be improved.  
In the notebook `project_analysis.ipynb` we make many visualizations and analysis on the given datasets and model in order to identify the most important metrics to be used to evaluate the performances of the model.

## 