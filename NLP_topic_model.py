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
