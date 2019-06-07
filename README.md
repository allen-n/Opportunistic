# CS205 Final Project - Spring 2019

## Allen's Notes:
### Documents:
* Final report linked [here](CS205_Final_AllenNikka.pdf)

## Initial Approach Plan:
### Feature Selection and Evaluation:
* Implement a new data-loader that will iterate through entire data set in batches and perform PCA on the batches, saving indices of top X% variance features
* Initial idea: select an additional 30 features that _could_ be relevant
* Justify features based on data supplied by the CDC [here](https://gis.cdc.gov/Cancer/USCS/DataViz.html).
* Implement PCA to prune ~50 features selected in above step

### Preprocessing & Feature Engineering:
* Think about after step 1, dependant on selected features

### Predictive Modeling:
* Try several different weak learners to gauge efficacy of each
* Tune hyperparameters on most promising weak learner
* Look at ensemble methods (Adaboost, etc.) to see if predictive power can be increased 
* Look at more complex models (NN, CNN, etc.)
___

## The Task:
In this project we will try and build a predicitve model to predict being positive to cancer or malignancy.
- Using the features available at [CDC NHANES](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2015), investigate what features you would like to use in your model. 
- Under the tab "Data, Documentation, Codebooks, SAS Code" choose one of the categories: "Demographics Data", "Dietary Data", "Examination Data", "Laboratory Data", or "Questionnaire Data". 
- For each of these categories there is a list of sub-categories available. Clicking the "Doc File" link will provide the information about features in that sub-category and feature names.
- Using these features, construct a predictive model that will be able to predict having cancer or malignancy.

**Example finding feature**: Our target of focus (cancer or malignancy) is found under Questionnaire Data -> Medical Conditions -> MCQ220 - Ever told you had cancer or malignancy. Under this question the variable name **MCQ220** is what we will use as an identifier in our code.

**Tip**: If you are looking for a feature, another option is to google "NHANES \<feature name\>". This is effective if you have a feature in mind but don't want to go over all possible categories to find it.

**Important**: When selecting features to process, remember to input the correct category where the feature can be found, so that the processing code can find it.
  
### What you are scored on
The grade for this task will be seperated into 3 categories
- **Feature selection, and evaluation**. You will be required to explain why certain features were incorporated. empiric evaluation (e.g: mutual information) or a logical explanation for a group of features (e.g: age, geneder are included as demographics, as certain types of cancer affect certain demographics with higher probability. As can be seen [here](https://gis.cdc.gov/Cancer/USCS/DataViz.html)).
- **Preprocessing and feature engineering**. Writing your own preprocessing code, explain why you chose a specific imputation technique, or changed the features in a certain way. You can support you argument with previous research done on the topic or with your own experimentation.
- **Predictive modeling**. Build an ML model to predict having cancer. Construct a model to fit the data, explain why the chosen model was selected, describe experiments done with the model (e.g: hyper parameter tuning).

### How you are scored
The grade will be given based on the quality, supported arguments, and clarity of evaluation of each of the 3 parts.
Each part is equal 33.3333...% of your final project grade. Bonus points for originality and "out of the box" thinking for approaching each of these 3 parts.


## File list:
- **nhanes.py:** implementation of the data preprocessing logic as well as definition an example dataset.
- **Demo_Dataset.ipynb:** Jupyter notebook file to demonstrate the basic usage of sample dataset.

## How to use:
1) Download [raw data files](https://drive.google.com/file/d/1hFp7O747408D8t5442f0Sjit7wXKXI1z/view?usp=sharing) and decompress them.
2) Install Python 3 and the following packages: joblib, numpy, pandas, matplotlib, scipy, sklearn, jupyter, pytorch.
3) Use Demo_Dataset.ipynb to see an example on how to use the predefined task.
4) Expand nhanes.py to define new tasks by following the implementation logic of the provided sample.


##### For a detailed explanation of the methods used here for the cost-sensetive health dataset, please refer to: ["Nutrition and Health Data for Cost-Sensitive Learning"](https://arxiv.org/abs/1902.07102)
