#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv(r"C:\Users\91989\Downloads\Mental-Health-Twitter.csv\Mental-Health-Twitter.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.isnull().sum() # check for missing values


# In[5]:


# keep only the required columns
df = df[['post_text']]
df.head()


# In[6]:


df.describe()


# In[7]:


df.shape


# In[8]:


get_ipython().system('pip install textblob')
get_ipython().system('pip install nltk')

from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word


# In[9]:


import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
sw = stopwords.words("english")
df["post_text"] = df["post_text"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))


# In[10]:


get_ipython().system('pip install textblob')
from textblob import TextBlob


# In[11]:


get_ipython().system('unzip /usr/share/nltk_data/corpora/wordnet.zip -d /usr/share/nltk_data/corpora/')


# In[12]:


# Lemmatization (to group similar words together)
from textblob import Word
nltk.download("wordnet")
nltk.download("omw-1.4")
df["post_text"] = df["post_text"].apply(lambda x: " ".join([Word(x).lemmatize()]))


# In[13]:


get_ipython().system('pip install spacy')
get_ipython().system('python -m spacy download en_core_web_sm')
get_ipython().system('pip install -U NLTK')


# In[14]:


# tokenize each word
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
df["tokens"] = df["post_text"].apply(lambda x: TextBlob(x).words)


# In[15]:


df.head()


# In[16]:


# Applying sentiment to entire dataset

blob_emptylist = []

for i in df["post_text"]:
    blob = TextBlob(i).sentiment # returns polarity
    blob_emptylist.append(blob)


# In[17]:


# Create a new dataframe to show polarity and subjectivity for each tweet
df2 = pd.DataFrame(blob_emptylist)
df2.head()


# In[18]:


# Combine both df and df2
df3 = pd.concat([df.reset_index(drop=True), df2], axis=1)
df3.head()


# In[19]:


# we only care about Positive or Negative hence drop subjectivity and only look at polarity
df4 = df3[['post_text','tokens','polarity']]
df4.head(6)


# In[20]:


# Sentiment value
df4["Sentiment"] =  np.where(df4["polarity"] >= 0 , "Positive", "Negative")
df4.head()


# In[21]:


result = df4["Sentiment"].value_counts()

sns.set_style("whitegrid")
sns.despine(left=True, bottom=True)
result.plot(kind="bar", rot=0, color=["plum","cyan"]);


# In[22]:


df4.groupby("Sentiment").count()


# In[23]:


df4.groupby("polarity").max().head(5)
# returns the tweets with maximum polarity i.e. most negative tweets


# In[24]:


#Visualize distribution of polarity
plt.figure(figsize=(8,4))
sns.histplot(df4['polarity'], bins=15, kde=False)
plt.xlabel('Polarity')
plt.ylabel('Count')
plt.title('Polarity Distribution')


# In[25]:


# Visualize distribution of sentiment
plt.figure(figsize=(10,6))
sns.countplot(x='Sentiment', data=df4,order=df4['Sentiment'].value_counts().index)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Distribution')
plt.show()


# # KNN
# 

# In[26]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[27]:


# split the data into test and train set
X_train, X_test, y_train, y_test = train_test_split(df4['post_text'], df4['Sentiment'], test_size=0.2, random_state=42)


# In[28]:


# Convert the text data into numerical features using a CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


# In[29]:


# create a KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)


# In[30]:


# Evaluate the classifier on the testing set
accuracy = knn.score(X_test, y_test)
print('Accuracy:', accuracy)


# In[31]:


# Get the accuracy score of the model
print('The accuracy of the KNN Classifier is',round(accuracy_score(knn.predict(X_test), y_test)*100,2),'%')


# In[32]:


# Create a classification report
print(classification_report(y_test, knn.predict(X_test)))


# In[33]:


# Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_test, knn.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot()


# In[ ]:




