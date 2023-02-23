# %% [markdown]
# # Import necessary libraries

# %%
import pandas as pd

# import warnings
# warnings.filterwarnings("ignore")

# %% [markdown]
# # Import Dataset

# %%
df = pd.read_csv('diabetes_data_upload.csv')
df.head()

# %% [markdown]
# # Dataset Preprocessing

# %%
df.columns

# %%
df

# %% [markdown]
# # Turning each factor and convert it into binary operators

# %%
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for n in df.columns:
    if n != "Age":
        le.fit(df[n])
        df[n] = le.transform(df[n])


# %% [markdown]
# # Reading the correlation between the features and its correlation to its result

# %%
df.corr(method='pearson')

# %% [markdown]
# # Splitting dataset

# %%
# from scipy.stats import mode
# from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# df.corr(method = 'pearson')

features = df.drop(columns=['class']).values
target = df['class'].values
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=12, stratify=target)

# features_train, features_test, target_train, target_test

# %% [markdown]
# # Reduce the model
# 
# Reducing the model with anything that has an absolute value less than 0.25

# %%
df2 = df
foo = df2.corr(method = 'pearson')
for i in foo:
    corr_to_class = foo['class'][i]
    if (abs(corr_to_class) < 0.25):
        df2 = df2.drop([i], axis=1)
df2

# %% [markdown]
# # Split the reduced dataset

# %%
features2 = df2.drop(columns=['class']).values
target2 = df['class'].values
features_train2, features_test2, target_train2, target_test2 = train_test_split(features2, target2, test_size=0.2, random_state=12, stratify=target2)

# features_train2, features_test2, target_train2, target_test2

# %% [markdown]
# # Model 1: Decision Tree

# %%
from sklearn import tree

# Regular model
model1 = tree.DecisionTreeClassifier(criterion='entropy')
model1.fit(features_train, target_train)


# Reduced model
r_model1 = tree.DecisionTreeClassifier(criterion='entropy')
r_model1.fit(features_train2, target_train2)

# %%
import pickle
filename = 'finalized_model_M1.model' 
pickle.dump(model1, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
r1_result = loaded_model.score(features_test, target_test)
print(r1_result)


# %%
filename = 'finalized_reduced_model_M1.model' 
pickle.dump(r_model1, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
r1r_result = loaded_model.score(features_test2, target_test2)
print(r1r_result)

# %% [markdown]
# # Model 2: Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression
model2=LogisticRegression()
r_model2=LogisticRegression()
model2.fit(features_train,target_train)
r_model2.fit(features_train2, target_train2)
print("Train_score    :",model2.score(features_train,target_train)*100)
print("Test_score     :",model2.score(features_test,target_test)*100)
print("Reduced Model")
print("Train_score    :",r_model2.score(features_train2,target_train2)*100)
print("Test_score     :",r_model2.score(features_test2,target_test2)*100)

# %%
filename = 'finalized_model_M2.model' 
pickle.dump(model2, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
r2_result = loaded_model.score(features_test, target_test)
print(r2_result)

# %%
filename = 'finalized_reduced_model_M2.model' 
pickle.dump(r_model2, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
r2r_result = loaded_model.score(features_test2, target_test2)
print(r2r_result)

# %% [markdown]
# # Model 3: Naive Bayes

# %%
from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(features_train,target_train)
r_model3 = GaussianNB()
r_model3.fit(features_train2,target_train2)

# %%
filename = 'finalized_model_M3.model' 
pickle.dump(model3, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
r3_result = loaded_model.score(features_test, target_test)
print(r3_result)

# %%
filename = 'finalized_reduced_model_M3.model' 
pickle.dump(r_model3, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
r3r_result = loaded_model.score(features_test2, target_test2)
print(r3r_result)

# %% [markdown]
# # Model 4: Random Forest

# %%
from sklearn.ensemble import RandomForestClassifier
model4 = RandomForestClassifier(max_depth=7, random_state=0)
model4.fit(features_train,target_train)
r_model4 = RandomForestClassifier(max_depth=7, random_state=0)
r_model4.fit(features_train2,target_train2)

# %%
filename = 'finalized_model_M4.model' 
pickle.dump(model4, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
r4_result = loaded_model.score(features_test, target_test)
print(r4_result)

# %%
filename = 'finalized_reduced_model_M4.model' 
pickle.dump(r_model4, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
r4r_result = loaded_model.score(features_test2, target_test2)
print(r4r_result)

# %% [markdown]
# # Evaluation

# %%
import matplotlib.pyplot as mp
data = [["Decision Tree", r1_result, r1r_result], ["Logistic Regression", r2_result, r2r_result], ["Naive Bayes", r3_result, r3r_result], ["Random Forest", r4_result, r4r_result]]
df = pd.DataFrame(data, columns=['Model Type', 'Complete Dataset', 'Reduced Dataset'])
df.plot(x="Model Type", y=["Complete Dataset", "Reduced Dataset"], kind="bar")
mp.legend(loc='lower left')

# %% [markdown]
# ## Analysis
# From this graph we are able to see that Random Forest has the best results with an accuraccy of 96.15%. We were supprised that reducing the features did not seem to improve the models except for Naive Bayes in which it increased the accuracy by about 3%. The following are the classification reports and confusion matrices for each model.

# %% [markdown]
# ### Decision Tree

# %%
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_predict = model1.predict(features_test)
print(classification_report(target_test, y_predict))

confusion_matrix = confusion_matrix(target_test, y_predict)
cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

# %% [markdown]
# ### Reduced Features Decision tree

# %%
from sklearn.metrics import classification_report, confusion_matrix
y_predict = r_model1.predict(features_test2)
print(classification_report(target_test2, y_predict))

confusion_matrix = confusion_matrix(target_test, y_predict)
cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()


# %% [markdown]
# ### Logistic Regression

# %%
from sklearn.metrics import classification_report, confusion_matrix
y_predict = model2.predict(features_test)
print(classification_report(target_test, y_predict))

confusion_matrix = confusion_matrix(target_test, y_predict)
cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

# %% [markdown]
# ### Reduced Features Logistic Regression

# %%
from sklearn.metrics import classification_report, confusion_matrix
y_predict = r_model2.predict(features_test2)
print(classification_report(target_test2, y_predict))

confusion_matrix = confusion_matrix(target_test, y_predict)
cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

# %% [markdown]
# ### Naive Bayes

# %%
from sklearn.metrics import classification_report, confusion_matrix
y_predict = model3.predict(features_test)
print(classification_report(target_test, y_predict))

confusion_matrix = confusion_matrix(target_test, y_predict)
cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

# %% [markdown]
# This is our worst performing model with 7 false negatives and 9 false positives.

# %% [markdown]
# ### Reduced Features Naive Bayes

# %%
from sklearn.metrics import classification_report, confusion_matrix
y_predict = r_model3.predict(features_test2)
print(classification_report(target_test2, y_predict))

confusion_matrix = confusion_matrix(target_test, y_predict)
cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

# %% [markdown]
# ### Random Forest

# %%
from sklearn.metrics import classification_report, confusion_matrix
y_predict = model4.predict(features_test)
print(classification_report(target_test, y_predict))

confusion_matrix = confusion_matrix(target_test, y_predict)
cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

# %% [markdown]
# This is the best performing model with only 1 false negative and 3 false positives.

# %% [markdown]
# ### Reduced Features Random Forest

# %%
from sklearn.metrics import classification_report, confusion_matrix
y_predict = r_model4.predict(features_test2)
print(classification_report(target_test2, y_predict))

confusion_matrix = confusion_matrix(target_test, y_predict)
cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()


