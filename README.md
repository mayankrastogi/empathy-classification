# Empathy Classification

Classification of people into empathetic and non-empathetic using the Young People Survey dataset.

---

This is a mini-project created as part of the **CS 412 - Introduction to Machine Learning** course, taught by *Professor Elena Zheleva*, at the University of Illinois at Chicago.

## Problem Statement

You are working for a non-profit that is recruiting student volunteers to help with Alzheimer's patients. You have been tasked with predicting how suitable a person is for this task by predicting how empathetic he or she is. Using the Young People Survey dataset (https://www.kaggle.com/miroslavsabo/young-people-survey/), predict a person's "Empathy" as either "very empathetic" (answers 4 and 5) or "not very empathetic" (answers 1, 2, and 3). You can use any of the other attributes in the dataset to make this prediction; however, you should not handpick the predictive features but let an algorithm select them. 

## Dependencies

Please make sure that you have the below packages installed on your system. Higher versions of these packages should also work, however, if you have trouble running the project, try installing the same versions as noted below. 

1. numpy 1.15.1
2. pandas 0.23.4
3. scikit-learn 0.20.1
4. joblib 0.13.0

## How to run?

1. Ensure that the mentioned packages in the dependencies section are installed on your system
2. Download and extract the project
3. Download the [Young People Survey dataset](https://www.kaggle.com/miroslavsabo/young-people-survey/) from Kaggle
4. Extract `responses.csv` and place it in a directory named `data` in the root directory of the project

#### Method 1: From Command-line

1. Open `command-prompt` (if on Windows) or `terminal` (if on Linux/Mac) on your system
2. Browse to the project directory
3. To run the project in `train` mode, issue the following command:

   `python main.py --mode=train --dataset=data/responses.csv --model=data/bestModel.pkl` 
4. To run the project in `test` mode, issue the following command:

   `python main.py --mode=test --dataset=data/testSet.csv --model=data/bestModel.pkl`

#### Method 2: From Jupyter Notebook

1. Launch Jupyter Notebook
2. Browse to project folder and open `hw5.ipynb`
3. Execute first cell to import the project's python file
4. To run the project in `train` mode, execute the second cell
5. To run the project in `test` mode, execute the third cell

## Write-up

The one page write-up file is named `hw5_writeup.pdf`, and can be found in the root directory of the project

## Extra-credit

The notebook for extra-credits, named `hw5_extra-credit.ipynb`, can be found in the root directory of the project.