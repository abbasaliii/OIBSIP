#!/usr/bin/env python
# coding: utf-8

# ## EMAIL SPAM DETECTION WITH MACHINE LEARNING

# ## Author: Syed Abbas Ali

# ### Import necessary libraries

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ### Load and Prepare the Data

# In[2]:


data = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\spam.csv", encoding='latin-1')
data.head(5)


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# In[6]:


data=data.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
data.head(5)


# In[7]:


data=data.rename(columns={"v1": "Label", "v2": "Text"})
data.head(5)


# In[8]:


data.duplicated().sum()


# In[9]:


data=data.drop_duplicates(keep="first")
data.head(5)


# In[10]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()

data["Label"]=encoder.fit_transform(data["Label"])
data.head(5)


# ### Split the dataset into training and testing sets

# In[11]:


x=data["Text"]
y=data["Label"]

X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)


# ### Text preprocessing and feature extraction using TF-IDF

# In[12]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer=TfidfVectorizer()
X_train_tfidf=tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf=tfidf_vectorizer.transform(X_test)


# ### Model Selection and Training

# In[13]:


nb_classifier = MultinomialNB()
# Hyperparameter tuning using GridSearchCV
param_grid = {
    'alpha': [0.1, 0.01, 0.001, 0.0001],  
}
grid_search = GridSearchCV(nb_classifier, param_grid, cv=5)
grid_search.fit(X_train_tfidf, y_train)
best_alpha = grid_search.best_params_['alpha']
best_nb_classifier = MultinomialNB(alpha=best_alpha)
best_nb_classifier.fit(X_train_tfidf, y_train)


# ### Model Evaluation

# In[14]:


# Evaluate the model on the test data
y_pred = best_nb_classifier.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[15]:


# Generate the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Generate the confusion matrix
confusion_matrix_result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_matrix_result)


# In[16]:


# Function to predict whether an email is spam or ham
def predict_spam(text):
    text_tfidf = tfidf_vectorizer.transform([text])
    prediction = best_nb_classifier.predict(text_tfidf)[0]
    return "spam" if prediction == 1 else "ham"

# Example usage:
text_to_predict = "I'm leaving my house now..."
result = predict_spam(text_to_predict)
print("Prediction:", result)


# In[17]:


# Visualize the confusion matrix
sns.heatmap(confusion_matrix_result, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


# In[ ]:




