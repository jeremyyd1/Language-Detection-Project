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

#Function to tokenize and clean the data

def text2sentences(text):
       sentence_list = []
       sentences_tokenized = sent_tokenize(text)
       for sentence in sentences_tokenized:
              sentence =  re.sub(r'[^\w\s]','', sentence)
              sentence =  re.sub(r'\n',' ', sentence)
              sentence =  re.sub(r' +',' ', sentence)
              sentence_list.append(sentence)

#returns a list of cleaned sentences
       return sentence_list
       
#Performs cleaning on sample texts
spanish_tokens_cleaned = text2sentences(spanish_text)
portuguese_tokens_cleaned = text2sentences(portuguese_text)

#Creates an array that contains labeled sentences 
X = np.array(spanish_tokens_cleaned + portuguese_tokens_cleaned)
y = np.array(['ESP']*len(spanish_tokens_cleaned) + ['PT']*len(portuguese_tokens_cleaned))


#Split data into training and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)

#Vectorization using char bigrams
vectorizer = CountVectorizer(analyzer= 'char', ngram_range=(2,2))

pipeline = Pipeline([
   ('vectorizer',vectorizer),  
   ('model',MultinomialNB())
])

#Fits the pipeline and calculates predictions based on the test set
pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)


while True:
       user_input = input("Enter a sentence to find out what language it's in!\n")
       if "exit" in user_input:
              print("Exiting...")
              break
       else:
              user_input = [user_input]
              pred = pipeline.predict(user_input)
              print("                 ")
              print("Sentence entered: {}".format(user_input))
              print("Predicted language: {}".format(pred))
              print("-----------------------")
