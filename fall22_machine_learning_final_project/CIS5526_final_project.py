"""
CIS 5526 final project

real vs fake news prediction
comparative study of datasets, vectorizers, and classifiers
"""

import pandas as pd
import numpy as np
import seaborn as sns
import gensim
from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional


### data preprocessing and EDA

def process_df(df):
    # remove rows containing missing values
    df.dropna(inplace=True)
    # combine all text
    df['alltext'] = df['title'] + ' ' + df['text']
    # tokenize and remove stop words
    df['p_alltext'] = df['alltext'].apply(lambda T : [t for t in simple_preprocess(T) if t not in gensim.parsing.preprocessing.STOPWORDS])
    # convert lists of words to strings
    df['sp_alltext'] = df['p_alltext'].apply(lambda l: " ".join(l))
    # shuffle examples
    df = df.sample(frac=1)
    return df

# dataset 1 - EDA conducted
true = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")
true # 21417 rows × 4 columns
fake # 23481 rows × 4 columns
# both all objects
true.dtypes
fake.dtypes
# label and combine dfs
true['label'], fake['label'] = 1, 0
news1 = pd.concat([true,fake]).reset_index(drop=True)

# dataset 2
# https://www.kaggle.com/datasets/hassanamin/textdb3
news2 = pd.read_csv('fake_or_real_news.csv')
news2['label'] = news2['label'].apply(lambda y: 1 if y == 'REAL' else 0)

# dataset 3
# https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
news3 = pd.read_csv('WELFake_Dataset.csv')
# its labels are coded in reverse relative to other datasets
news3['label'] = news3['label'].apply(lambda y: 1 if y == 0 else 0)

# dataframe list
dfl = [process_df(df) for df in [news1,news2,news3]]
# split data sets: list of three tuples of X_train, X_test, y_train, y_test for each df
sds = [tuple(train_test_split(dfl[j].sp_alltext,np.asarray(dfl[j].label))) for j in range(3)]


## EDA for dataset 1
# much of these ideas from the following source
# https://www.kaggle.com/code/paramarthasengupta/fake-news-detector-eda-prediction-99

# word clouds
plt.figure()
realnewswords = " ".join([" ".join(l) for l in dfl[0][dfl[0].label==1].p_alltext])
wc_real = WordCloud(max_words=1000,width=1000,height=1000,background_color='black',colormap='winter').generate(realnewswords)
plt.imshow(wc_real)
wc_real.to_file('real_news_wordcloud.png')
fakenewswords = " ".join([" ".join(l) for l in dfl[0][dfl[0].label==0].p_alltext])
wc_fake = WordCloud(max_words=1000,width=1000,height=1000,background_color='white',colormap='autumn').generate(fakenewswords)
plt.imshow(wc_fake)
wc_fake.to_file('fake_news_wordcloud.png')
plt.clf()
plt.close()

# histogram of word counts
fig, ax1 = plt.subplots()
ax1.set_title("Word count frequency across real and fake news articles")
ax1.hist([len(l) for l in dfl[0][dfl[0].label==0].p_text],bins=50,label="fake news",alpha=0.6,color='r')
ax1.hist([len(l) for l in dfl[0][dfl[0].label==1].p_text],bins=50,label="real news",alpha=0.6,color='b')
# plt.hist([len(l) for l in news.p_alltext],bins=50,label="real news",alpha=0.6,color='g')
ax1.legend()
ax1.set_xlabel("word count")
ax1.set_ylabel("frequency")
ax2 = fig.add_axes([0.4, 0.2, 0.45, 0.5])
ax2.hist([len(l) for l in dfl[0][dfl[0].label==0].p_text],bins=50,label="fake news",alpha=0.6,color='r')
ax2.hist([len(l) for l in dfl[0][dfl[0].label==1].p_text],bins=50,label="real news",alpha=0.6,color='b')
ax2.set_xlim(0,1000)
plt.show()
fig.savefig("word_count_hist.png")

# counts of articles by subject
dfl[0].subject = dfl[0].subject.apply(lambda s: s.lower().replace('-',' ')).replace({'politics':'politics news','politicsnews':'politics news'})
rnvs = [list(dfl[0][dfl[0].label==1].subject).count(c) for c in dfl[0].subject.unique()]
fnvs = [list(dfl[0][dfl[0].label==0].subject).count(c) for c in dfl[0].subject.unique()]
x = np.arange(len(rnvs))
fig,ax = plt.subplots()
b1 = ax.barh(x-0.3/2,rnvs,0.3,label="real news",color='b')
b2 = ax.barh(x+0.3/2,fnvs,0.3,label='fake news',color='r')
ax.set_title("Subjects of real and fake news")
ax.set_ylabel("subject")
ax.set_yticks(x)
ax.set_yticklabels(list(dfl[0].subject.unique()))
ax.set_xlabel("count")
ax.legend()
plt.show()
fig.savefig("subject_barh.png")


### LSTM model

# data dict
lstmadd = {j:[] for j in range(3)}
# history list
hl = []

# iterate over 3 datasets once for fitting and again for testing
for j in range(3):
    # create and fit tokenizer
    nuw = len({l[i] for l in dfl[j].p_alltext for i in range(len(l))})
    tok = Tokenizer(num_words=nuw)
    tok.fit_on_texts(sds[j][0])
    pad = max([len(l) for l in dfl[j].p_alltext])
    # tokenize and pad training and testing sequences
    xtTs = [tuple(tok.texts_to_sequences(sds[q][i]) for i in range(2)) for q in range(3)]
    pxtTs = [(pad_sequences(xtTs[q][0],maxlen=pad,padding='post',truncating='post'),pad_sequences(xtTs[q][1],maxlen=pad,padding='post')) for q in range(3)]
    # build model - this part is directly from https://www.kaggle.com/code/sanchukanirupama/lstm-based-fake-news-detection
    lstm_model = Sequential()
    lstm_model.add(Embedding(nuw,output_dim=256))
    lstm_model.add(Bidirectional(LSTM(128)))
    lstm_model.add(Dense(128, activation='relu'))
    lstm_model.add(Dense(64, activation='relu'))
    lstm_model.add(Dense(1, activation='sigmoid'))
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    lstm_model.summary()
    # train model
    h = lstm_model.fit(pxtTs[j][0], sds[j][2], batch_size=64, validation_split=0.1, epochs=1)
    hl.append(h)
    # test model
    for k in range(3):
        lstmadd[j].append(accuracy_score(sds[k][3],[1 if y > 0.5 else 0 for y in lstm_model.predict(pxtTs[k][1])]))
# accuracy data and graph
lstmadf = pd.DataFrame.from_dict(lstmadd)
lstmadf.to_excel("acc_scores_lstm.xlsx")
ll = ['Dataset {}'.format(j) for j in range(1,4)]
fig,ax = plt.subplots()
sns.heatmap(lstmadf,cmap='gnuplot2',annot=True,xticklabels=ll,yticklabels=ll,fmt='.2%',ax=ax)
plt.title('Accuracy scores of LSTM models across 3 datasets')
plt.xlabel('training')
plt.ylabel('testing')
plt.savefig('acc_scores_lstm.jpeg',bbox_inches='tight')

# # saving last model because it took the longest
# lstm_model.save('D3LSTM')
# tm = load_model('D3LSTM')
# tm.summary()


### 5 non-NN classifiers

# split data sets: list of three tuples of X_train, X_test, y_train, y_test for each df
# data did not need to be split again, but in practice this ended up being the best option
# ideally, everything would be rerun on the same split, but that is a considerable amount of time and computer power for an altogether noncritical consistency
sds = [tuple(train_test_split(dfl[j].sp_alltext,np.asarray(dfl[j].label))) for j in range(3)]

# classifier and data dicts
cfd = {'RF':RandomForestClassifier(),'GNB':GaussianNB(),'LSVC':LinearSVC(),'kNN':KNeighborsClassifier(7),'LR':LogisticRegression()}
c5add = {k:[] for k in cfd.keys()}
# iterate over 3 datasets once for fitting, 2 vectorizers, 5 classifiers, and 3 datasets again for testing
for j in range(3):
    for v in [CountVectorizer(max_features=1000),TfidfVectorizer(max_features=1000)]:
        # fit vectorizer to X_train - was v.fit(dfl[j].alltext)
        v.fit(sds[j][0])
        # transform X train/test portions
        txtTs = [tuple(v.transform(sds[q][i]) for i in range(2)) for q in range(3)]
        # fit models and predict
        for cf in cfd.items():
            cf[1].fit(txtTs[j][0].toarray(),sds[j][2])
            for k in range(3):
                c5add[cf[0]].append(accuracy_score(sds[k][3],cf[1].predict(txtTs[k][1].toarray())))
# accuracy data and heatmap
c5addf = pd.DataFrame.from_dict(c5add)
c5addf.to_excel("acc_scores_5_clfs.xlsx")
rls = ['{}-{}-{}'.format(Dt,V,DT) for Dt in range(1,4) for V in ['c','t'] for DT in range(1,4)]
fig,ax = plt.subplots()
sns.heatmap(c5addf,cmap='gnuplot2',annot=True,xticklabels=list(cfd.keys()),yticklabels=rls,fmt='.2%',ax=ax)
plt.title('Accuracy scores across classifiers, vectorizers, and datasets')
plt.savefig('acc_scores_5_clfs.png',bbox_inches='tight')


# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
# https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/pad_sequences
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

# dataset info
pd.DataFrame({k:[df.label.value_counts()[0],df.label.value_counts()[1],df.shape[0],len({l[i] for l in df.p_alltext for i in range(len(l))}),max([len(l) for l in df.p_alltext])] for k,df in zip(ll,dfl)},['fake examples','real examples','total examples','unique words','max length']).to_excel('datasetinfo.xlsx')

"""
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_1 (Embedding)     (None, None, 256)         29341952  
                                                                 
 bidirectional_1 (Bidirectio  (None, 256)              394240    
 nal)                                                            
                                                                 
 dense_3 (Dense)             (None, 128)               32896     
                                                                 
 dense_4 (Dense)             (None, 64)                8256      
                                                                 
 dense_5 (Dense)             (None, 1)                 65        
                                                                 
=================================================================
Total params: 29,777,409
Trainable params: 29,777,409
Non-trainable params: 0
_________________________________________________________________
474/474 [==============================] - 3319s 7s/step - loss: 0.0372 - acc: 0.9861 - val_loss: 0.0042 - val_acc: 0.9991
351/351 [==============================] - 393s 1s/step
50/50 [==============================] - 56s 1s/step
559/559 [==============================] - 597s 1s/step
Model: "sequential_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_2 (Embedding)     (None, None, 256)         16718592  
                                                                 
 bidirectional_2 (Bidirectio  (None, 256)              394240    
 nal)                                                            
                                                                 
 dense_6 (Dense)             (None, 128)               32896     
                                                                 
 dense_7 (Dense)             (None, 64)                8256      
                                                                 
 dense_8 (Dense)             (None, 1)                 65        
                                                                 
=================================================================
Total params: 17,154,049
Trainable params: 17,154,049
Non-trainable params: 0
_________________________________________________________________
67/67 [==============================] - 942s 14s/step - loss: 0.4047 - acc: 0.8000 - val_loss: 0.2432 - val_acc: 0.8992
351/351 [==============================] - 657s 2s/step
50/50 [==============================] - 94s 2s/step
559/559 [==============================] - 1046s 2s/step
Model: "sequential_4"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_3 (Embedding)     (None, None, 256)         59232768  
                                                                 
 bidirectional_3 (Bidirectio  (None, 256)              394240    
 nal)                                                            
                                                                 
 dense_9 (Dense)             (None, 128)               32896     
                                                                 
 dense_10 (Dense)            (None, 64)                8256      
                                                                 
 dense_11 (Dense)            (None, 1)                 65        
                                                                 
=================================================================
Total params: 59,668,225
Trainable params: 59,668,225
Non-trainable params: 0
_________________________________________________________________
755/755 [==============================] - 33336s 44s/step - loss: 0.1478 - acc: 0.9431 - val_loss: 0.1047 - val_acc: 0.9609
351/351 [==============================] - 1517s 4s/step
50/50 [==============================] - 214s 4s/step
559/559 [==============================] - 2409s 4s/step
"""
