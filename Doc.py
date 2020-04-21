#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libraries
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import unidecode
import unicodedata
import re
import ftfy
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec


# In[64]:


# Import Data
df = pd.read_csv('C:\\Users\\PRUNEDGE\\Downloads\\doc_summaries.csv', encoding = 'Windows-1252')
 
# Check for null values
df[df.isnull().any(axis=1)]
 
# Drop rows with null Values
df.drop(df[df.isnull().any(axis=1)].index,inplace=True)
#df.drop(columns = 'id', inplace = True)
df.head()


# In[66]:


df['tag'] = df.index
df = df.reset_index()
df = df.drop(['index'], axis = 1)


# In[3]:


df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split()])) 
df['text'] = df['text'].replace(u'\u201c', '"').replace(u'\u201d', '"')


# In[4]:


prev = df['text'].values.tolist()

processed_features = []
for sentence in range(0, len(prev)):
    # Remove all the special characters
    clean = ftfy.fix_text(str(prev[sentence]), uncurl_quotes = True)
    processed_feature = re.sub('[\W_]+', ' ', clean)
     # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
     # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
    #Removing special characters and sequences
    processed_feature = re.sub(r'\“', '', processed_feature)
    processed_feature = re.sub(r'\”', '', processed_feature)
    processed_feature = re.sub(r'\_', '', processed_feature)
    processed_feature = re.sub(r",", " ", processed_feature )
    processed_feature = re.sub(r"\.", " ", processed_feature )  
    processed_feature = re.sub(r"!", " ", processed_feature )  
    processed_feature = re.sub(r"\(", " ( ", processed_feature )
    processed_feature = re.sub(r"\)", " ) ", processed_feature )
    processed_feature = re.sub(r"\?", " ", processed_feature )
    processed_feature = re.sub(r"\s{2,}", " ", processed_feature )
    processed_feature = re.sub(r"[^A-Za-z0-9(),!.?\'`\"\“\”]", " ", processed_feature )
     # Converting to Lowercase
    processed_feature = processed_feature.lower()
    processed_features.append(processed_feature)


# In[73]:


#Creating document list of list   
train_docs = list(map(lambda el:[el], processed_features))

#Tagging document sentences 
comp_docs = [TaggedDocument(
                words=[word for word in document[0].lower().split()],
                tags = [i]
            ) for i, document in enumerate(train_docs)]


# In[48]:


max_epoch = 2
vec_size = 20
    
# Train model
model = Doc2Vec(size = vec_size, dm = 0, dbow_words = 1, window = 2, alpha = 0.2)
model.build_vocab(comp_docs)
for epoch in range(max_epoch):
    model.train(comp_docs, total_examples = model.corpus_count, epochs = epoch)
    
model.save("Doc2Vec.model")
print("Model Saved")


# In[78]:


def build_model(test_doc, compiled_doc):
    '''
    Parameters
    -----------
    test_doc: list of lists - one of the sentence lists
    compiled_doc: list of lists - combined sentence lists to match the index to the sentence 
    '''
    model= Doc2Vec.load("C:/Users/PRUNEDGE/Downloads/tixati/Doc2Vec.model")
    scores = []
    #for doc in test_docs:
    dd = {}
    # Calculate the cosine similarity and return top 25 matches
    score = model.docvecs.most_similar([model.infer_vector(test_docs)],topn=25)
    key = " ".join(test_docs)
    for i in range(len(score)):
    # Get index and score
        x, y = score[i]
        z = df['id'].loc[df.index == x].values[0]
    # Match sentence from other list
        nkey = ' '.join(comp_docs[x][0])
        dd[z, nkey] = y
    scores.append({key:dd})

    return scores


# In[75]:


#Enter test document here 
test_docs = word_tokenize('authorized to take such actions as instructed by any effective date indenture dated as of among hovnanian enterprises by their acceptance of the notes shall be deemed to have instructed and authorized to take such action as instructed by any effective date grant pari passu on such property to for the benefit of the')


# In[79]:


#Run model 
build_model(test_docs, comp_docs)

