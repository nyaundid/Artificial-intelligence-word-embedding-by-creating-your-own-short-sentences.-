# coding: utf-8

# In[65]:


from textblob import TextBlob
from gensim.models import Word2Vec
# define training data
sentences = [['this' 'is' 'the' 'first' 'sentence'], ['this' 'is' 'the' 'second' 'sentence'], ['the' 'cat' 'in' 'the' 'hat'],
      ['my' 'dog' 'is' 'mad'], ['i' 'hate' 'animals'], ['i' 'like' 'money'], ['who' 'is' 'the' 'best'],[' when' 'do' 'i' 'leave'],
      ['my' 'name' 'is' 'jim'], ['i' 'live' 'there'], ['when' 'do' 'we' 'leave'], ['keep' 'it' 'up'], ['the' 'bathroom' 'is' 'clean'],
      ['dont' 'be' 'rude'], ['have' 'fun'], ['dance' 'like' 'you' 'cant'], ['four' 'trees' 'in' 'the' 'park'], ['i' 'am' 'tall'],
      ['i' 'am' 'fast'], ['go' 'there' 'now']]


# In[67]:


import numpy as np
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
# define training data

# train model
model = Word2Vec(sentences, min_count=1)
# fit a TSNE model to the vectors
X = model[model.wv.vocab]
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
# create a scatter plot of the projection
# create a scatter plot of the projection
pyplot.scatter(X_tsne[:, 0], X_tsne[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(X_tsne[i, 0], X_tsne[i, 1]))
pyplot.show()
import time
import datetime
print (datetime.datetime.now().strftime("%y-%m-%d-%H-%M"))

