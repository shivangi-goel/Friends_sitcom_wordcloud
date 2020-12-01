# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 23:00:07 2020

@author: DELL
"""
import re
import unicodedata
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import numpy as np
from PIL import Image
import wordcloud
from wordcloud import ImageColorGenerator
from wordcloud import WordCloud
import matplotlib.pyplot as plt

path="C:\\Users\\DELL\\Desktop\\Study Material\\courses\\SNU Semesters\\SEM_7\\CSD350 NLP\\transcript.txt"

#getting 
def get_dialogs(actor):
    output = []          
    with open(path, 'r') as f:
        for line in f:
            if re.findall(r'(^'+actor+r'.*:.*)',line,re.IGNORECASE):
                output.append(line)
    f.close()
    return output

def word_expansion(phrase):

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def text_cleaning(text):
    #removing character's name
    text=re.sub(r'.*:', '', text)
    #print(text)
    #removing all types of brackets
    text=re.sub('[\(\[].*?[\)\]]', ' ', text)
    #print(text)
    #remove accents and normalization of text
    text=unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')    
    #print(text)
    #case folding
    text=text.lower()
    #print(text)
    #removing punctuation
    text = re.sub(r'[^\w\s]', '', text)
    #print(text)
    return text


def create_token(text):
    #removing stop words
    stopword_list = stopwords.words('english')    
    tokens = nltk.word_tokenize(text) 
    #print(tokens)
    tokens = [token.strip() for token in tokens]
    #print(tokens)
    tokens=' '.join([token for token in tokens if token not in stopword_list])
    #print(tokens)
    #lemmatization
    lemmatizer=WordNetLemmatizer()
    tokens= lemmatizer.lemmatize(tokens)
    #print(tokens)
    return tokens

query=input("Enter the friends character name you want wordcloud for.\n")

#removal of specific words.
extrawords=['never', 'like', 'this', 'that', 'okay','gon', 'think', 'ok', 'would', 'should', 'will', 'shall', 'can', 'could', 'oh', 'know']
def create_text(query):
    dialog1=get_dialogs(str(query))

    #text cleaning 
    #text="Monica: hello there, how are you(crying)? Rachel... you're there?"


    #text=word_expansion(text)
    #print(text)

    """    
    text=text_cleaning(text)
    tokens=[]
    tokens=create_token(text)
    print(tokens)
    """

    final1=[]
    for i in dialog1:
        token=[]
        exp=word_expansion(i)
        clean=text_cleaning(exp)
        token=create_token(clean)
        final1.append(token)
    rem_word=[word for word in final1 if word.lower() not in extrawords]
    actor_text=""
    for i in rem_word:
        actor_text+=i
        #print(monica_text) 
    
    return actor_text


#print(create_text(query))


def create_word_cloud(image_path, monica_text):
    #creating image mask    
    
    char_mask = np.array(Image.open(image_path))    
    image_colors = ImageColorGenerator(char_mask)

    wc = WordCloud(background_color="black", max_words=100, width=500, height=600, mask=char_mask, random_state=1).generate(monica_text)
    # to recolour the image
    plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")


image_path="C:\\Users\\DELL\\Desktop\\Study Material\\courses\\SNU Semesters\\SEM_7\\CSD350 NLP\\images\\"
if(query=='monica'):
    image_path=image_path+"monica.jpg" 
    monica_text=create_text(query)
    create_word_cloud(image_path, monica_text)
    
elif(query=='chandler'):
    image_path=image_path+"chandler.jpg" 
    chandler_text=create_text(query)
    create_word_cloud(image_path, chandler_text)

elif(query=='ross'):
    image_path=image_path+"ross.jpg" 
    ross_text=create_text(query)
    create_word_cloud(image_path, ross_text)
    
elif(query=='rachel'):
    image_path=image_path+"rachel.jpg" 
    rachel_text=create_text(query)
    create_word_cloud(image_path, rachel_text)
    
elif(query=='phoebe'):
    image_path=image_path+"phoebe.jpg" 
    phoebe_text=create_text(query)
    create_word_cloud(image_path, phoebe_text)
    
elif(query=='joey'):
    image_path=image_path+"joey.jpg" 
    joey_text=create_text(query)
    create_word_cloud(image_path, joey_text)
    
mon_text=create_text("monica")
ross_text=create_text("ross")
rach_text=create_text("rachel")
joey_text=create_text("joey")
ph_text=create_text("phoebe")
chan_text=create_text("chandler")

#top 10 frequent words list

#finding frequency
def word_freq(text):
    token=text.split()
    freq=[token.count(w) for w in token]
    freq_list=list(zip(token, freq))
    return freq_list


mon_freq=word_freq(mon_text)
rach_freq=word_freq(rach_text)
ph_freq=word_freq(ph_text)
chan_freq=word_freq(chan_text)
ross_freq=word_freq(ross_text)
joey_freq=word_freq(joey_text)

#print(mon_freq)

#frequency count using counter:
from collections import Counter 

def top_10_count(text):
    token=text.split()
    count = Counter(token)
    top_10_words=count.most_common(10)
    return top_10_words

mon10=top_10_count(mon_text)
chan10=top_10_count(chan_text)
rach10=top_10_count(rach_text)
ross10=top_10_count(ross_text)
ph10=top_10_count(ph_text)
joey10=top_10_count(joey_text)

print("10 frequently used words by Monica are: \n", mon10)
print("10 frequently used words by Chandler are: \n", chan10)
print("10 frequently used words by Rachel are: \n", rach10)
print("10 frequently used words by Ross are: \n", ross10)
print("10 frequently used words by Phoebe are: \n", ph10)
print("10 frequently used words by Joey are: \n", joey10)

#classification

#creating dataset
def dataset(act):
    text1=get_dialogs(act)
    list1=[]
    for i in text1:
        i=re.sub(r'.*:', '', i)
        i=re.sub('[\(\[].*?[\)\]]', ' ', i)
        #edit1+=i
        list1.append(i)

    list2=[]

    for i in list1:
        i = re.sub(r'[^\w\s]', '', i) 
        list2.append(i)
    return list2

mon_list=dataset("monica")
rach_list=dataset("rachel")
ross_list=dataset("ross")
joey_list=dataset("joey")
ph_list=dataset("phoebe")
chan_list=dataset("chan")

#length of lists
m=len(mon_list)
ra=len(rach_list)
ro=len(ross_list)
p=len(ph_list)
c=len(chan_list)
j=len(joey_list)

text=[]
actor=[]
text=mon_list+rach_list+ross_list+joey_list+ph_list+chan_list
for i in range(m):
    actor.append("Monica")
for i in range(m, m+ra):
    actor.append("Rachel")
for i in range(m+ra,m+ra+ro):
    actor.append("Ross")
for i in range(m+ra+ro, m+ra+ro+j):
    actor.append("Joey")
for i in range(m+ra+ro+j, m+ra+ro+j+p):
    actor.append("Phoebe")
for i in range(m+ra+ro+j+p, m+ra+ro+j+p+c):
    actor.append("Chandler")


import pandas as pd
d={'Text':text, 'Actor':actor}
df=pd.DataFrame(d)

#working on dataset
from sklearn import preprocessing 
label_encoder=preprocessing.LabelEncoder()
df['Actor']=label_encoder.fit_transform(df['Actor'])

"""
Encoding-
Chandler-0
Joey-1
Monica-2
Phoebe-3
Rachel-4
Ross-5
"""
def guess_char(Y_pred):
    if(Y_pred==0):
        print("Chandler")
    elif(Y_pred==1):
        print("Joey")
    elif(Y_pred==2):
        print("Monica")
    elif(Y_pred==3):
        print("Phoebe")
    elif(Y_pred==4):
        print("Rachel")
    elif(Y_pred==5):
        print("Ross")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

X=df['Text']
Y=df['Actor']

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.30, random_state=42)

#naive bayes with Tf-idf
vectorizer=TfidfVectorizer()

x=input("Enter the sentence to be tested\n")
X_test=[x]
X_train_tf = vectorizer.fit_transform(X_train)
X_test_tf = vectorizer.transform(X_test)

classifier = MultinomialNB()
model1=classifier.fit(X_train_tf, Y_train)

Y_pred = model1.predict(X_test_tf)

guess_char(Y_pred)
#acc = 1-metrics.accuracy_score(Y_test, Y_pred)
#accuracy of this model is around 72%

#decision tree
from sklearn.tree import DecisionTreeClassifier

X_train_tf = vectorizer.fit_transform(X_train)
X_test_tf = vectorizer.transform(X_test)

classifier2=DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=5, min_samples_leaf=8) 
model2=classifier2.fit(X_train_tf, Y_train)
Y_pred2=model2.predict(X_test_tf) 
guess_char(Y_pred2)
#acc2=1-metrics.accuracy_score(Y_test, Y_pred2)

#accuracy of this model is around 79%


#logistic regression
from sklearn.linear_model import LogisticRegression
classifier3=LogisticRegression(random_state = 0) 
model3=classifier3.fit(X_train_tf, Y_train)
Y_pred3=model3.predict(X_test_tf) 
#acc3=1-metrics.accuracy_score(Y_test, Y_pred3)   

#accuracy of this model is around 72% 
guess_char(Y_pred3)         

