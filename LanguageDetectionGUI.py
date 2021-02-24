import numpy as np
import nltk
import re
import tkinter as tk
from tkinter import *
from nltk import sent_tokenize
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from text_data import spanish_text, portuguese_text


#Function to strip the brackets and quotation marks from the prediction
def strip_pred(pred):
    pred = re.sub(r'[^\w]', '', pred)
    return pred

#Function to get input, predict language, and then print it to the screen
def predict_lang():
    lang_prediction = tk.Label(root, text="                                         ") #Replaces current label
    lang_prediction.grid(row=5, column=0)
    lang_prediction.config(font=('verdana', 12))

    user_input = entry_main.get()
    entry_main.delete(0, END)
    user_input = [user_input]
    pred = pipeline.predict(user_input)
    pred = str(pred)
    pred = strip_pred(pred)
    lang_prediction = tk.Label(root, text="Language: " + pred)
    lang_prediction.grid(row=5, column=0)
    lang_prediction.config(font=('verdana', 12))


#Configures the GUI 
root = tk.Tk()
root.geometry("600x300")
root.title("Language Detection Project")


header_main = tk.Label(root, text='Language Detector (Spanish or Portuguese)')
header_main.config(font=('verdana', 16))
header_main.grid(row=0, column=0, columnspan=2, pady=20)

entry_main_label = tk.Label(root, text="Enter a sentence below to find out if it's in Spanish or Portuguese: ")
entry_main_label.grid(row=2, column=0, pady=10)
entry_main_label.config(font=('verdana', 12))

entry_main = tk.Entry(root, width=60)
entry_main.grid(row=3,column=0, ipady=10, pady=10)

submit_btn = tk.Button(root, text="Submit", command=predict_lang)
submit_btn.config(font=('helvetica', 12))
submit_btn.grid(row=4, column=0, ipady=10, pady=10)



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
y = np.array(['Spanish']*len(spanish_tokens_cleaned) + ['Portuguese']*len(portuguese_tokens_cleaned))


#Split data into training and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=42)

#Vectorization using char bigrams
vectorizer = CountVectorizer(analyzer= 'char', ngram_range=(2,2))

pipeline = Pipeline([
   ('vectorizer',vectorizer),  
   ('model',MultinomialNB())
])

#Fits the pipeline and calculates predictions based on the test set
pipeline.fit(X_train,y_train)




root.mainloop()
