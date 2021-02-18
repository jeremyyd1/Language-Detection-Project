import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re
from nltk import sent_tokenize
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix,classification_report
from text_data import spanish_text, portuguese_text



def text2sentences(text):
       sentence_list = []
       sentences_tokenized = sent_tokenize(text)
       for sentence in sentences_tokenized:
              sentence =  re.sub(r'[^\w\s]','', sentence)
              sentence =  re.sub(r'\n',' ', sentence)
              sentence =  re.sub(r' +',' ', sentence)
              sentence_list.append(sentence)

       
       return sentence_list
       

spanish_tokens_cleaned = text2sentences(spanish_text)
portuguese_tokens_cleaned = text2sentences(portuguese_text)

X = np.array(spanish_tokens_cleaned + portuguese_tokens_cleaned)
y = np.array(['ESP']*len(spanish_tokens_cleaned) + ['PT']*len(portuguese_tokens_cleaned))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)

vectorizer = CountVectorizer(analyzer= 'char', ngram_range=(2,2))

pipeline = Pipeline([
   ('vectorizer',vectorizer),  
   ('model',MultinomialNB())
])

pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))