<!-- ABOUT THE PROJECT -->
## About The Project

XAIplayground is a Python library that facilitates the creation and comparison of explainable classification models. 

This library contains four different modules, `DataExplainability`, `DataPreprocessing`, `DataProcessing`, and `ModelExplainability`. The modules can be used independently, or combined to follow the workflow from the dataset to explainable AI.

DataExplainability : handles the exploration part of the workflow, by creating different plots in order to visualize the data.

DataPreprocess : pre-processes the dataset by splitting the features from the taget, encoding the categorical columns, scaling the features and reducing the dimensionality of the dataset.

DataProcessing : handles the "processing" of the data, by traning the models and subsequently validating them. 

ModelExplainability : creates local and global explanations for the model's predictions.


### Built With

* python 3.9.3
* numpy 1.20.3
* pandas 1.2.4
* scikit-learn 0.24.2
* lime 0.2.0.1

For a the full dependencies list, read the `requirements.txt` 


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running in a virtual environment follow these simple steps:

### Installation

1. `git clone <repo>`
2. `cd <repo>`
3. `pip install virtualenv` 
4. `virtualenv venv` 
5. `venv/bin/activate` 
6. `pip install -r requirements.txt`



<!-- USAGE EXAMPLES -->
## Usage

A project tutorial can be found in /pipeline/Tutorial.ipynb

A short tutorial on the plots can be found in /pipeline/Plots.ipynb

<!-- CONTACT -->
## Contact

José Andrés Cordova - jose.cordova@uni-duesseldorf.de

Project Link: [https://gitlab.cs.uni-duesseldorf.de/dbs/students/bachelor/ba_jose_andres_cordova](https://gitlab.cs.uni-duesseldorf.de/dbs/students/bachelor/ba_jose_andres_cordova)



