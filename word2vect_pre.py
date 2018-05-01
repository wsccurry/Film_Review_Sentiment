import nltk.data
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')

unlabels_train=pd.read_csv('unlabeledTrainData.tsv',delimiter='\t',quoting=3)

def review_to_text(review,remove_stopwords):
    raw_text=BeautifulSoup(review,'html').get_text()
    letters=re.sub('[^a-zA-Z]',' ',raw_text)
    words=letters.lower().split()
    if remove_stopwords:
        stop_words=set(stopwords.words('english'))
        words=[w for w in words if w not in stop_words]
    return words

def review_to_sentences(review,tokenizer):
    raw_sentences=tokenizer.tokenize(review.strip())
    sentences=[]
    for raw_sentence in raw_sentences:
        if len(raw_sentence)>0:
            sentences.append(review_to_text(raw_sentence))
    return sentences

corpora=[]

for review in unlabels_train['review']:
    corpora+=review_to_sentences(review,tokenizer)

num_features=300
min_word_count=20
num_workers=4
context=10
downsampling = 1e-3

from gensim.models import word2vec

model=word2vec.Word2Vec(corpora,workers=num_workers,size=num_features,min_count=min_word_count,window=context,sample=downsampling)

model.init_sims(replace=True)

model_name="IMDB_Sentiment/300features_20minwords_10context"

model.save(model_name)

model=word2vec.Word2Vec.load('IMDB_Sentiment/300features_20minwords_10context')

print(model.most_similar('man'))

import numpy as np


#用一个评论中的每个词的词向量做平均法得到一个评论段落的向量
def makeFeatures(words,model,num_features):
    featureVec=np.zeros((num_features,),dtype=np.float32)
    nwords=0.
    index2word_set=set(model.index2word)
    for word in words:
        if word in index2word_set:
            nwords+=1
            featureVec=np.add(featureVec,model[word])
    featureVec=np.divide(featureVec,nwords)
    return featureVec

def getAvgFeatureVects(reviews,model,num_features):
    counter=0
    reviewFeatureVecs=np.zeros((len(reviews),num_features),dtype=np.float32)
    for review in reviews:
        reviewFeatureVecs[counter]=makeFeatures(review,model,num_features)
        counter+=1
    return reviewFeatureVecs

train=pd.read_csv('labeledTrainData.tsv',delimiter='\t')
y_train=train['sentiment']
test=pd.read_csv('testData.tsv',delimiter='\t')

clean_train_reviews=[]
for review in train['review']:
    clean_train_reviews.append(review_to_text(review,remove_stopwords=True))


trainDataVects=getAvgFeatureVects(clean_train_reviews,model,num_features)

clean_test_reviews=[]

for review in test['review']:
    clean_train_reviews.append(review_to_text(review,remove_stopwords=True))

testDataVecs=getAvgFeatureVects(clean_test_reviews,model,num_features)

from sklearn.ensemble import  GradientBoostingClassifier

gbc=GradientBoostingClassifier(n_estimators=500,learning_rate=0.1,max_depth=4)

gbc.fit(trainDataVects,y_train)

result=gbc.predict(testDataVecs)

output=pd.DataFrame({'id':test[id],'sentiment':result})

output.to_csv('submission.csv')