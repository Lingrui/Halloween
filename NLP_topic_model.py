#!/usr/bin/python 
#import packages 
import base64
import numpy as np
import pandas as pd
import plotly.offline as py
#py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from collections import Counter
from scipy.misc import imread
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from matplotlib import pyplot as plt
#%matplotlib inline

# Loading in the training data with Pandas
train = pd.read_csv("./input/train.csv")

#take a look at a quick peek of waht the first three rows in the data 
print(train.head())

#how large the training data is 
print(train.shape)
'''
z = {'EAP': 'Edgar Allen Poe', 'MWS': 'Mary Shelley', 'HPL': 'HP Lovecraft'}
data = [go.Bar(
            x = train.author.map(z).unique(),
            y = train.author.value_counts().values,
            marker= dict(colorscale='Jet',
                         color = train.author.value_counts().values
                        ),
            text='Text entries attributed to Author'
    )]

layout = go.Layout(
    title='Target variable distribution'
)

fig = go.Figure(data=data, layout=layout)

#py.iplot(fig, filename='basic-bar')
py.plot(fig,filename='basic-bar')


eap_words = train['text'].str.split(expand=True).unstack().value_counts()
data = [go.Bar(
            x = eap_words.index.values[2:50],
            y = eap_words.values[2:50],
            marker= dict(colorscale='Jet',
                         color = eap_words.values[2:100]
                        ),
            text='Word counts'
    )]

layout = go.Layout(
    title='Top 50 (Uncleaned) Word frequencies'
)

fig = go.Figure(data=data, layout=layout)

py.plot(fig, filename='word_freq')

#WordClouds to visualise each author's work
eap = train[train.author=="EAP"]["text"].values
hpl = train[train.author=="HPL"]["text"].values
mws = train[train.author=="MWS"]["text"].values

from wordcloud import WordCloud, STOPWORDS
import codecs
# Generate the Mask for EAP
f1 = open("eap.png", "wb")
f1.write(codecs.decode(eap_64,'base64'))
f1.close()
img1 = imread("eap.png")
# img = img.resize((980,1080))
hcmask = img1

f2 = open("mws.png", "wb")
f2.write(codecs.decode(mws_64,'base64'))
f2.close()
img2 = imread("mws.png")
hcmask2 = img2

f3 = open("hpl.png", "wb")
f3.write(codecs.decode(hpl_64,'base64'))
f3.close()
img3 = imread("hpl.png")
hcmask3 = img3
'''

#storing the first text element as a string
import nltk
#nltk.download('punkt')
#Tokenization
first_text = train.text.values[0]
print (first_text)
print ('='*90)
print(first_text.split(' '))

first_text_list = nltk.word_tokenize(first_text)
print ('='*90)
print(first_text_list)

#Stopword removal
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
print(len(stopwords))
print(stopwords)

first_text_list_cleaned = [word for word in first_text_list if word.lower() not in stopwords]
print(first_text_list_cleaned)
print("="*90)
print("Length of original list: {0} words\n"
      "Length of list after stopwords removal: {1} words"
      .format(len(first_text_list), len(first_text_list_cleaned)))
	
#Stemming
stemmer = nltk.stem.PorterStemmer()
print("The stemmed form of running is: {}".format(stemmer.stem("running")))
print("The stemmed form of runs is: {}".format(stemmer.stem("runs")))
print("The stemmed form of run is: {}".format(stemmer.stem("run")))

#Vectorizing raw text 
#defining our sentence
sentence = ["I love to eat Burgers",
			"I love to eat Fries"]
vectorizer = CountVectorizer(min_df=0)
sentence_transform = vectorizer.fit_transform(sentence)

print ("\nThe features are:\n {}".format(vectorizer.get_feature_names()))
print ("\nThe vectorized array looks like:\n {}".format(sentence_transform.toarray()))
