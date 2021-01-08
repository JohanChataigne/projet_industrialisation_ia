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

## Performance Tests
 It is important to know the performance of your service. In order to evaluate the REST API implemented previously, we choose to use Locust, which works well with Flask.
 We will make a given number of simultaneous calls to the API to test the average response time:
 
<img src="test-img/biglocust.png" alt="test" width="700"/>
 
 As we can see, the service has some difficulties answering to a number of users greater than a dozen. Even with 15 users at a time, the API takes up to 3 minutes to anwser. After a moment, it crashes down. 
 It surely isn't ready to be put online according to these tests.
 
### Scaling

This test was made on a Intel Core i5-9600KF CPU at 3.70Ghz. To scale our system vertically, we could try to use the GPU in order to calculate faster the answers. In fact, the computer was having a hard time trying to handle even a pack of 5 users at a time. Scaling vertically is the most cost-efficient method to improve response time and stability. We could also use a higher amout of devices to run the API.

### Load ramp-up test

To load ramp-up test our API, we launch Locust with different parameters. The aim is to have an approximative maximum number of simultaneous users our service can handle.

To do so, we add a user every 100 seconds.

<img src="test-img/loadrampup.png" alt="ramp-up" width="700"/>

We can see with this test that 7 users at a time seems to be the limit. To improve precision, we made 10 other tests with similar parameters and ended with a mean of a maximum of 8 users at a time.

With this test, we can also see that the approximate response time is : 

`number_of_simultaneous_users * 10 seconds.`
