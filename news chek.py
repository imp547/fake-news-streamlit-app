# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:59:13 2025

@author: mk
"""

#import libaries

import pandas as pd
import  re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

#load data 
true_news = pd.read_csv("archive/True.csv")
fake_news = pd.read_csv("archive/Fake.csv")


#Encoding
true_news['label']=1
fake_news['label']=0

#combine dataset 
data= pd.concat([true_news,fake_news], ignore_index= True)

#shuffle dataset
data = data.sample(frac=1,random_state=42).reset_index(drop=True)

#info
data.info()
data.head(10)


#download source
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

#lemmentization
lemmantizer= WordNetLemmatizer()
stop_word=set(stopwords.words('english'))

def preprocess_text(text):
    text=text.casefold()
    text=re.sub(r'[^a-z\s]','', text)
    tokens=word_tokenize(text)
    tokens=[lemmantizer.lemmatize(word) for word in tokens if word not in stop_word]
    return ' '.join(tokens)

#applying
data['cleaned_text']=data['text'].apply(preprocess_text)

data['cleaned_text'].head()


#vectoriser
tfidf = TfidfVectorizer(max_features=5000)

#fit and transform TF-IDF
x=tfidf.fit_transform(data['cleaned_text']).toarray()
y=data['label']


print("\nShape of TF-IDF Matrix:", x.shape)

#train test split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#modle building 
model= LogisticRegression(max_iter=1000,random_state=42)

model.fit(x_train,y_train)

y_pred =model.predict(x_test)


print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

#fitcheck
train_accuracy = model.score(x_train, y_train)
test_accuracy = model.score(x_test, y_test)

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Testing Accuracy: {test_accuracy:.2f}")


#saving model

with open("fake_news_model.pkl","wb") as file:
    pickle.dump(model, file)
    
with open("tfidf_vectorizer.pkl","wb") as file:
    pickle.dump(tfidf,file)
    