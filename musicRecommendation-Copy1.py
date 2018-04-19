
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
#data=pd.read_csv(r"C:\Users\yvonn\OneDrive\Documents\WeChat Files\yifan729796133\Files\writeNewDataLog_2018-03-16\writeNewDataLog_2018-03-16.csv",sep='\t',encoding="utf-8")
#data1=pd.read_csv(r"C:\Users\yvonn\OneDrive\Documents\WeChat Files\yifan729796133\Files\writeNewDataLog20180315\writeNewDataLog20180315.csv",sep='\t',encoding="utf-8")
data1=pd.read_csv(r"C:\Users\yvonn\Downloads\writeNewDataLog_2018-03-19\writeNewDataLog_2018-03-19.csv",sep='\t',encoding="utf-8",usecols=['user_id_x','user_work_id','activity','score'])
data2=pd.read_csv(r"C:\Users\yvonn\Downloads\writeNewDataLog_2018-03-19\writeNewDataLog_2018-03-27.csv",sep='\t',encoding="utf-8",usecols=['user_id_x','user_work_id','activity','score'])
data3=pd.read_csv(r"C:\Users\yvonn\Downloads\writeNewDataLog_2018-03-19\writeNewDataLog_2018-03-28.csv",sep='\t',encoding="utf-8",usecols=['user_id_x','user_work_id','activity','score'])
data4=pd.read_csv(r"C:\Users\yvonn\Downloads\writeNewDataLog_2018-03-19\writeNewDataLog_2018-03-29.csv",sep='\t',encoding="utf-8",usecols=['user_id_x','user_work_id','activity','score'])
data5=pd.read_csv(r"C:\Users\yvonn\Downloads\writeNewDataLog_2018-03-19\writeNewDataLog_2018-03-30.csv",sep='\t',encoding="utf-8",usecols=['user_id_x','user_work_id','activity','score'])

temp=pd.concat([data4,data1,data2,data3,data5],ignore_index=True)


# In[2]:


len(temp.user_id_x.drop_duplicates())


# In[3]:


len(temp.user_work_id.drop_duplicates())


# In[15]:


#送礼 12，评论 10，分享 8，下载 6 收藏 4， 播放 2
temp.loc[temp.activity==u'送礼物','score']=12 ##
temp.loc[temp.activity==u'评论','score']=10
temp.loc[temp.activity==u'播放','score']=2
temp.loc[temp.activity==u'下载','score']=6


# In[16]:


data=temp.groupby(['user_id_x','user_work_id'],as_index=False).sum()
#data=data.drop(['user_id_y','song_id','singer_id'],axis=1)
data=data.groupby('user_id_x').filter(lambda x:len(x)>10)
data=data.groupby('user_work_id').filter(lambda x:len(x)>10)


# In[100]:


len(temp)


# In[9]:


temp['activitytime'].max()


# In[12]:


#data.columns=['user_id', 'item_id', 'rating']
df=data


# In[ ]:


df.user_id_x=df.user_id_x.rank(method='dense')
df.user_work_id=df.user_work_id.rank(method='dense')
df=df.astype(int)


# In[32]:


df


# In[33]:


n_users = df.user_id_x.unique().shape[0]
n_items = df.user_work_id.unique().shape[0]
print((str(n_users) + ' users'))
print((str(n_items) + ' items'))

ratings = np.zeros((n_users, n_items))
for row in df.itertuples():
    ratings[row[1]-1, row[2]-1] = row[3]
print(ratings)

sparsity = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0] * ratings.shape[1])
sparsity *= 100
print(('Sparsity: {:4.2f}%'.format(sparsity)))


# In[9]:


#(test!=0).sum()


# In[34]:


def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0],
                                        size=10)
                                        #replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]

    # Test and training are truly disjoint
    assert(np.all((train * test) == 0))
    return train, test

train, test = train_test_split(ratings)


# In[35]:


def slow_similarity(ratings, kind='user'):
    if kind == 'user':
        axmax = 0
        axmin = 1
    elif kind == 'item':
        axmax = 1
        axmin = 0
    sim = np.zeros((ratings.shape[axmax], ratings.shape[axmax]))
    for u in range(ratings.shape[axmax]):
        for uprime in range(ratings.shape[axmax]):
            rui_sqrd = 0.
            ruprimei_sqrd = 0.
            for i in range(ratings.shape[axmin]):
                sim[u, uprime] = ratings[u, i] * ratings[uprime, i]
                rui_sqrd += ratings[u, i] ** 2
                ruprimei_sqrd += ratings[uprime, i] ** 2
            sim[u, uprime] /= rui_sqrd * ruprimei_sqrd
    return sim


# In[36]:


def fast_similarity(ratings, kind='user', epsilon=1e-9):
    # epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)


# In[37]:


user_similarity = fast_similarity(train, kind='user')
item_similarity = fast_similarity(train, kind='item')
print((item_similarity[:4, :4]))


# In[38]:


def predict_slow_simple(ratings, similarity, kind='user'):
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        for i in range(ratings.shape[0]):
            for j in range(ratings.shape[1]):
                pred[i, j] = similarity[i, :].dot(ratings[:, j])                             /np.sum(np.abs(similarity[i, :]))
        return pred
    elif kind == 'item':
        for i in range(ratings.shape[0]):
            for j in range(ratings.shape[1]):
                pred[i, j] = similarity[j, :].dot(ratings[i, :].T)                             /np.sum(np.abs(similarity[j, :]))

        return pred


# In[39]:


def predict_fast_simple(ratings, similarity, kind='user'):
    if kind == 'user':
        return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif kind == 'item':
        return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])


# In[40]:


predict_user_train=predict_fast_simple(ratings,user_similarity,'user')
predict_item_train=predict_fast_simple(ratings,item_similarity,'item')


# In[41]:


predict_user_train1=predict_fast_simple(train,user_similarity,'user')
predict_item_train1=predict_fast_simple(train,item_similarity,'item')


# In[42]:


predict_user_train2=predict_fast_simple(test,user_similarity,'user')
predict_item_train2=predict_fast_simple(test,item_similarity,'item')


# In[43]:


temp1=ratings!=0
#temp1=temp1.replace(False,0)
#temp1=temp1.replace(True,1,inplace=aTrue)
rating2=temp1.astype(int)


# In[44]:


temp1=train!=0
#temp1=temp1.replace(False,0)
#temp1=temp1.replace(True,1,inplace=True)
train2=temp1.astype(int)


# In[48]:


temp1=test!=0
#temp1=temp1.replace(False,0)
#temp1=temp1.replace(True,1,inplace=True)
test2=temp1.astype(int)


# In[62]:


del temp1
del df


# In[97]:


import gc
gc.collect()


# In[70]:


a=predict_item_train*rating2


# In[95]:


b=a!=0
c=ratings!=0


# In[ ]:


from sklearn.metrics import mean_squared_error
mean_squared_error(predict_item_train*rating2,ratings)


# In[ ]:


abs(predict_item_train*rating2-ratings).sum()/rating2.sum()


# In[25]:


from sklearn.metrics import mean_squared_error
mean_squared_error(predict_user_train*rating2,ratings)


# In[158]:


abs(predict_user_train*rating2-ratings).sum()/rating2.sum()


# In[159]:


abs(predict_user_train1*train2-train).sum()/train2.sum()


# In[160]:


abs(predict_item_train1*train2-train).sum()/train2.sum()


# In[161]:


abs(predict_item_train2*test2-test).sum()/test2.sum()


# In[162]:


abs(predict_user_train2*test2-test).sum()/test2.sum()


# In[163]:


abs(train.sum()/train2.sum()*test2-test).sum()/test2.sum()


# In[208]:


#temp.dropna(how='any')


# In[167]:


temp1=temp[['user_id_x','gender_y','score']]
scoretable=temp1.groupby(['user_id_x','gender_y'],as_index=False).sum()
#temp1.groupby(['user_id_x','user_charact_x']).count()
traindata1=scoretable.pivot_table(values='score',index='user_id_x',columns='gender_y',aggfunc=np.sum).replace(np.nan,0)
temp2=temp[['user_id_x','song_lang','score']]
scoretable2=temp2.groupby(['user_id_x','song_lang'],as_index=False).sum()
temp3=temp[['user_id_x','song_genre','score']]
scoretable3=temp3.groupby(['user_id_x','song_genre'],as_index=False).sum()
traindata2=scoretable2.pivot_table(values='score',index='user_id_x',columns='song_lang',aggfunc=np.sum).replace(np.nan,0)
traindata3=scoretable3.pivot_table(values='score',index='user_id_x',columns='song_genre',aggfunc=np.sum).replace(np.nan,0)
temp4=temp[['user_id_x','user_age_y','score']]
scoretable4=temp4.groupby(['user_id_x','user_age_y'],as_index=False).sum()
traindata4=scoretable4.pivot_table(values='score',index='user_id_x',columns='user_age_y',aggfunc=np.sum).replace(np.nan,0)
ff=pd.concat([traindata1,traindata2,traindata3,traindata4],axis=1)
ff=ff.replace(np.nan,0)


# In[195]:


user2=ff[ff.index.isin(data.user_id)]


# In[196]:


user2=user2.as_matrix()


# In[197]:


user_similarity1 = fast_similarity(user2, kind='user')


# In[198]:


predict_user=predict_fast_simple(ratings,user_similarity1,'user')


# In[200]:


abs(predict_user*rating2-ratings).sum()/rating2.sum()


# In[232]:


temp1=temp[['user_id_x','gender_y','score']]
scoretable=temp1.groupby(['user_id_x','gender_y'],as_index=False).sum()
#temp1.groupby(['user_id_x','user_charact_x']).count()
traindata1=scoretable.pivot_table(values='score',index='user_id_x',columns='gender_y',aggfunc=np.sum).replace(np.nan,0)
temp2=temp[['user_id_x','song_lang','score']]
scoretable2=temp2.groupby(['user_id_x','song_lang'],as_index=False).sum()
temp3=temp[['user_id_x','song_genre','score']]
scoretable3=temp3.groupby(['user_id_x','song_genre'],as_index=False).sum()
traindata2=scoretable2.pivot_table(values='score',index='user_id_x',columns='song_lang',aggfunc=np.sum).replace(np.nan,0)
traindata3=scoretable3.pivot_table(values='score',index='user_id_x',columns='song_genre',aggfunc=np.sum).replace(np.nan,0)
temp4=temp[['user_id_x','user_age_y','score']]
scoretable4=temp4.groupby(['user_id_x','user_age_y'],as_index=False).sum()
traindata4=scoretable4.pivot_table(values='score',index='user_id_x',columns='user_age_y',aggfunc=np.sum).replace(np.nan,0)
temp5=temp[['user_id_x','region_x','score']]
scoretable5=temp5.groupby(['user_id_x','region_x'],as_index=False).sum()
traindata5=scoretable5.pivot_table(values='score',index='user_id_x',columns='region_x',aggfunc=np.sum).replace(np.nan,0)
ff=pd.concat([traindata1,traindata2,traindata3,traindata4, traindata5],axis=1)
ff=ff.replace(np.nan,0)


# In[234]:


user2=ff[ff.index.isin(data.user_id)]
user2=user2.as_matrix()
user_similarity1 = fast_similarity(user2, kind='user')
predict_user=predict_fast_simple(ratings,user_similarity1,'user')
abs(predict_user*rating2-ratings).sum()/rating2.sum()


# In[27]:


a=ratings-ratings.mean()


# In[28]:


ratings


# In[29]:


predict_user1=predict_fast_simple(a,user_similarity,'user')
predict_item1=predict_fast_simple(a,item_similarity,'item')


# In[30]:


temp1=a!=0
#temp1=temp1.replace(False,0)
#temp1=temp1.replace(True,1,inplace=aTrue)
rating2=temp1.astype(int)


# In[31]:


from sklearn.metrics import mean_squared_error
mean_squared_error(predict_item1*rating2,a)


# In[230]:


abs(predict_item1*rating2-a).sum()/rating2.sum()


# In[32]:


from sklearn.metrics import mean_squared_error
mean_squared_error(predict_user1*rating2,a)


# In[231]:


abs(predict_user1*rating2-a).sum()/rating2.sum()

