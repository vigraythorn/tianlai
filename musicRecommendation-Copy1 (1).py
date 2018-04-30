
# coding: utf-8

# In[2]:


import findspark
findspark.init()
import pyspark


# In[3]:


#import spark package
import pandas as pd
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
sc = SparkContext(appName="recommendation")
sql_sc = SQLContext(sc)
# Load and parse the data


# In[4]:


#data=pd.read_csv(r"C:\Users\yvonn\OneDrive\Documents\WeChat Files\yifan729796133\Files\writeNewDataLog_2018-03-16\writeNewDataLog_2018-03-16.csv",sep='\t',encoding="utf-8")
#data1=pd.read_csv(r"C:\Users\yvonn\OneDrive\Documents\WeChat Files\yifan729796133\Files\writeNewDataLog20180315\writeNewDataLog20180315.csv",sep='\t',encoding="utf-8")
#data1=pd.read_csv(r"\data\data\writeNewDataLog_2018-03-19.csv",sep='\t',encoding="utf-8",usecols=['user_id_x','user_work_id','activity','score'])
#data1 = sql_sc.createDataFrame(data1)

data21=pd.read_csv("/data/data/writeNewDataLog_2018-03-27.csv",sep='\t',encoding="utf-8",usecols=['user_id_x','user_work_id','activity'])
data2 = sql_sc.createDataFrame(data21)
#data31=pd.read_csv("/data/data/writeNewDataLog_2018-03-28.csv",sep='\t',encoding="utf-8",usecols=['user_id_x','user_work_id','activity','score'])
#data3 = sql_sc.createDataFrame(data31)
#data41=pd.read_csv("/data/data/writeNewDataLog_2018-03-29.csv",sep='\t',encoding="utf-8",usecols=['user_id_x','user_work_id','activity','score'])
#data4 = sql_sc.createDataFrame(data41)
#data5=pd.read_csv("/data/data/writeNewDataLog_2018-03-30.csv",sep='\t',encoding="utf-8",usecols=['user_id_x','user_work_id','activity','score'])
#data5 = sql_sc.createDataFrame(data5)


# In[5]:


temp=data2


# In[6]:


#temp=data2.union(data3)


# In[7]:


#temp=pd.concat([data4,data2,data3],ignore_index=True)


# In[8]:


#print out the columns
temp.printSchema()


# In[ ]:


#get distinct user_id and user_work_id
temp.select('user_id_x').distinct().count()


# In[ ]:


temp.select('user_work_id').distinct().count()


# In[9]:


#change dataframe to a table
temp.registerTempTable('df')


# In[10]:


#use sql command to deal with table defined
sqlContext = SQLContext(sc)
temp=sqlContext.sql("select *,(case when activity = '送礼物' then 12 when activity='评论' then 10 when activity='播放' then 2 when activity='下载' then 6 end) as score from df")


# In[31]:


sqlContext.dropTempTable("df")


# In[13]:


data=temp.groupby(['user_id_x','user_work_id']).agg({'score': 'sum'})
data.show(5)


# In[43]:


data = data.withColumnRenamed("sum(score)", "score")


# In[61]:


data.count()


# In[46]:


data.registerTempTable('data')
sqlContext.sql("select user_id_x,user_work_id,score from data").show(5)


# In[74]:


from pyspark.sql.functions import col
user=data.groupby('user_id_x').count().withColumnRenamed("count", "n").filter("n >= 6").select('user_id_x')
data1=data.join(user,['user_id_x'])
work=data1.groupby('user_work_id').count().withColumnRenamed("count", "n").filter("n >= 6").select('user_work_id')
data2=data1.join(work,['user_work_id'])


# In[73]:


data2.show(5)


# In[92]:


#data.columns=['user_id', 'item_id', 'rating']
df=data2
from pyspark.sql.functions import *
from pyspark.sql.window import Window

ranked =  df.withColumn("rank", dense_rank().over(Window.orderBy("user_id_x")))
ranked1=ranked.withColumn("rank1",dense_rank().over(Window.orderBy("user_work_id")))


# In[96]:


df=ranked1.select('rank','rank1','score').withColumnRenamed("rank", "user_id_x").withColumnRenamed("rank1","user_work_id",)


# In[88]:


ranked.select('rank').rdd.max()[0]


# In[97]:


df.show(5)


# In[108]:


n_users = df.select('user_id_x').distinct().count()
n_items = df.select('user_work_id').distinct().count()
print((str(n_users) + ' users'))
print((str(n_items) + ' items'))

ratings = np.zeros((n_users, n_items))
for row in df.rdd.collect():
    ratings[row[0]-1, row[1]-1] = row[2]
print(ratings)

sparsity = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0] * ratings.shape[1])
sparsity *= 100
print(('Sparsity: {:4.2f}%'.format(sparsity)))


# In[109]:


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


# In[110]:


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


# In[111]:


def fast_similarity(ratings, kind='user', epsilon=1e-9):
    # epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)


# In[112]:


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


# In[ ]:


from sklearn.metrics import mean_squared_error
mean_squared_error(predict_item_train*rating2,ratings)


# In[ ]:


abs(predict_item_train*rating2-ratings).sum()/rating2.sum()


# In[25]:



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


# In[30]:


temp1=a!=0
#temp1=temp1.replace(False,0)
#temp1=temp1.replace(True,1,inplace=aTrue)
rating2=temp1.astype(int)


# In[31]:


mean_squared_error(predict_item1*rating2,a)


# In[230]:


abs(predict_item1*rating2-a).sum()/rating2.sum()


# In[32]:


mean_squared_error(predict_user1*rating2,a)


# In[231]:


abs(predict_user1*rating2-a).sum()/rating2.sum()


# In[7]:


import scipy.sparse
import pandas as pd
df=pd.DataFrame([[0,0,0],[0,1,0],[0,1,0]])
a=scipy.sparse.csr_matrix(df.values)

