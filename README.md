<!-- ABOUT THE PROJECT -->
## About XAIplayground

XAIplayground is a Python library that facilitates the creation and comparison of explainable classification models. 

This library contains five different modules, `DataExplainability`, `DataPreprocessing`, `DataProcessing`, `ModelExplainability`, and `ModelPipeline`. The modules can be used independently, or combined to follow the workflow from the dataset to explainable AI.

DataExplainability : handles the exploration part of the workflow, by creating different plots in order to visualize the data.

DataPreprocess : pre-processes the dataset by splitting the features from the taget, encoding the categorical columns, scaling the features and reducing the dimensionality of the dataset.

DataProcessing : handles the "processing" of the data, by traning the models and subsequently validating them. 

ModelExplainability : creates local and global explanations for the model's predictions.

ModelPipeline : predicts data through the pipeline.


### Built With

* python 3.9.3
* numpy 1.20.3
* pandas 1.2.4
* scikit-learn 0.24.2
* lime 0.2.0.1
* shap 0.39.0

For a the full dependencies list, read the `requirements_win.txt` 


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running in a virtual environment follow these simple steps:

### Installation


1. `git clone https://gitlab.cs.uni-duesseldorf.de/dbs/students/bachelor/ba_jose_andres_cordova`
2. `cd /ba_jose_andres_cordova`
3. `pip install virtualenv` 
4. `virtualenv venv` 

Linux|Windows
-----|----
5\. run `venv/bin/activate`|5. run `venv/Scripts/activate` 
6\. `pip install -r requirements_linux.txt`|6. `pip install -r requirements_win.txt`

#### Troubleshooting



SHAP not installing on Linux : `sudo apt-get install python3-dev`.

<!-- USAGE EXAMPLES -->
## Usage

An example project can be found in /pipeline/AcademicPerformance.ipynb

An example of data exploration can be found in /pipeline/DataVisualization.ipynb

<!-- CONTACT -->
## Contact

José Andrés Cordova - jose.cordova@uni-duesseldorf.de

Project Link: [https://gitlab.cs.uni-duesseldorf.de/dbs/students/bachelor/ba_jose_andres_cordova](https://gitlab.cs.uni-duesseldorf.de/dbs/students/bachelor/ba_jose_andres_cordova)

<!-- LICENSE -->
## License

[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0//)

