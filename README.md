# CPSC-483-Machine-Learning-Diabetes-Recognition

## Objective
Build a Machine Learning model to predict the likelyhood of diabetes in individuals based on their responses to a questionairre.

## Description
Using the UCI dataset https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset., build a machine learning model to reduce risk in late diabetes diagnoses. The dataset comes from study by the Sylhet Diabetes Hospital in Sylhet, Bangladesh and it was conducted using questionnaires from the patients. The end use case for this machine learning project could be to incorporate the model into an online questionairre. Anyone could fill out the form and the model could recommend the user has a doctor visit if it predicts they are at risk of diabetes.

## Methods
The dataset will first be preprocessed, part of this process includes removing parts of the data that have a low correlation value of 25% or less to create a reduced dataset. The models will then be built, and there will be four types of models made in an effort to compare them and see which performs best. The models will also be made using the original dataset and the reduced dataset to further compare performance, the types of models made will be Logistic Regression, Naive Bayes, Decision Tree, and Random Forest.
