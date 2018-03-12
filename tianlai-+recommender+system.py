
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:
##read in the result sample data file

temp=pd.read_csv(r"C:\Users\yvonn\Downloads\result.csv",sep='\t',encoding="utf-8") 


# In[3]:


temp.columns ##show all the features of interest


# ## Split train and test

# In[5]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(temp,
                                   stratify=temp['user_id_x'], 
                                   test_size=0.20,
                                   random_state=42)
								   
##first method for generating user-user and item-item collaborative filtering	

							   
ratings_train=train[['user_work_id','user_id_x','score']]
ratings_test=test[['user_work_id','user_id_x','score']]

###building user-item matrix

train_data_matrix=ratings_train.groupby(['user_id_x','user_work_id']).sum().reset_index().pivot_table(values='score',index='user_work_id',columns='user_id_x',aggfunc=np.sum).replace(np.nan,0).T
test_data_matrix=ratings_test.groupby(['user_id_x','user_work_id']).sum().reset_index().pivot_table(values='score',index='user_work_id',columns='user_id_x',aggfunc=np.sum).replace(np.nan,0).T


# ## similarity between items and users

# In[7]:

## compute similarities between users and items

from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')


# ## Make prediction

# In[8]:

## Based on similarity of users and items. Compute score between each user and item.
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = ratings.sub(mean_user_rating,axis=0)
        pred1 = np.array(mean_user_rating) + similarity.dot(ratings_diff).T / np.abs(similarity).sum(axis=1)
        pred=pd.DataFrame(pred1.T)
    elif type == 'item':
        pred = ratings.dot(similarity) / np.abs(similarity).sum(axis=1)
    return pred


# In[242]:


#train_data_matrix.dot(item_similarity)/np.abs(item_similarity).sum(axis=1)


# In[9]:

## used predict function to compute each score in the user-item matrix
item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')


# In[135]:

## find 10 user works with largest 10 score values for each user based on item-item collaborative filtering. 
prediction_item=pd.DataFrame({n: item_prediction.T[col].nlargest(10).index.tolist() 
                  for n, col in enumerate(item_prediction.T)}).T
result_item_1=pd.DataFrame()
## find 'user_charact_y','song_lang','song_type' for each top 10 user works for each user
for i in range(100):
    rec=temp[temp['user_work_id'].isin(prediction_item.iloc[i])][['user_charact_y','song_lang','song_type']].drop_duplicates().apply(tuple, axis=1).reset_index().drop(['index'],axis=1).T
    result_item_1=result_item_1.append(rec)
result_item_1


# In[206]:
## find 10 user works with largest 10 score values for each user based on user-user collaborative filtering. 

prediction_user=pd.DataFrame({n: user_prediction.T[col].nlargest(10).index.tolist() 
                  for n, col in enumerate(user_prediction.T)}).T
result_user_1=pd.DataFrame()
## find 'user_charact_y','song_lang','song_type' for each top 10 user works for each user
for i in range(100):
    rec=temp[temp['user_work_id'].isin(prediction_user.iloc[i])][['user_charact_y','song_lang','song_type']].drop_duplicates().apply(tuple, axis=1).reset_index().drop(['index'],axis=1).T
    result_user_1=result_user_1.append(rec)
    
result_user_1


# # Test Accuracy
# ## first method for testing accuracy

# In[ ]:

## find 'user_charact_y','song_lang','song_type' for each user works for each user in test data set.
## find the overlap between test data set and our predictions(user-user one and item-item one)
testact=test.groupby('user_id_x').apply(lambda x:x[['user_charact_y','song_lang','song_type']].apply(tuple,axis=1)).reset_index()
num_item=[testact[testact['user_id_x']==i][0].isin(result_item_1.iloc[i]).sum() for i in range(100)]
num_user=[testact[testact['user_id_x']==i][0].isin(result_user_1.iloc[i]).sum() for i in range(100)]


# In[227]:

## result of test method 1
print sum(num_item),sum(num_user)


# ## second method for testing accuracy

# In[197]:

## find occurrence of 'user_charact_y','song_lang','song_type' in top 10 predictions.
## find the overlap between test data set and our predictions.
testact2=test.groupby('user_id_x').apply(lambda x:x[['user_id_x','user_charact_y','song_lang','song_type']])
num_user2=[]
for i in range(100):
    rec=temp[temp['user_work_id'].isin(prediction_user.iloc[i])][['user_charact_y','song_lang','song_type']].drop_duplicates().mode(axis=0)
    num_user2.append(testact2[testact2['user_id_x']==i][['user_charact_y','song_lang','song_type']].isin(rec.to_dict('list')).sum(axis=1).sum())
num_item2=[]
for i in range(100):
    rec=temp[temp['user_work_id'].isin(prediction_item.iloc[i])][['user_charact_y','song_lang','song_type']].drop_duplicates().mode(axis=0)
    num_item2.append(testact2[testact2['user_id_x']==i][['user_charact_y','song_lang','song_type']].isin(rec.to_dict('list')).sum(axis=1).sum())


# In[228]:

## overlap result for test method 2 
print sum(num_item2),sum(num_user2)


# ## baseline

# In[219]:
## Find the baseline for recommendation that is randomly recommend user works to users

def base():
    base=pd.DataFrame(np.random.randint(0,1000,size=(100, 10)))
    result_base_1=pd.DataFrame()
    for i in range(100):
        rec=temp[temp['user_work_id'].isin(base.iloc[i])][['user_charact_y','song_lang','song_type']].drop_duplicates().apply(tuple, axis=1).reset_index().drop(['index'],axis=1).T
        result_base_1=result_base_1.append(rec)

    basetestact=test.groupby('user_id_x').apply(lambda x:x[['user_charact_y','song_lang','song_type']].apply(tuple,axis=1)).reset_index()
    base_1=[basetestact[basetestact['user_id_x']==i][0].isin(result_base_1.iloc[i]).sum() for i in range(100)]
    basetestact2=test.groupby('user_id_x').apply(lambda x:x[['user_id_x','user_charact_y','song_lang','song_type']])
    base_2=[]
    for i in range(100):
        rec=temp[temp['user_work_id'].isin(base.iloc[i])][['user_charact_y','song_lang','song_type']].drop_duplicates().mode(axis=0)
        base_2.append(basetestact2[basetestact2['user_id_x']==i][['user_charact_y','song_lang','song_type']].isin(rec.to_dict('list')).sum(axis=1).sum())
    return [sum(base_1),sum(base_2)]


# In[232]:


base_result=pd.DataFrame([base() for i in range(100)])


# In[233]:


#base_result=base_result.mean(axis=0)
base_result.mean(axis=0)


# In[205]:



# ## user-user method2

# In[247]:

## method 2 user-user collaborative filtering
## create user to user-work feature matrix
temp1=train[['user_id_x','user_charact_y','score']]
scoretable=temp1.groupby(['user_id_x','user_charact_y']).sum().reset_index()
#temp1.groupby(['user_id_x','user_charact_x']).count()
traindata1=scoretable.pivot_table(values='score',index='user_id_x',columns='user_charact_y',aggfunc=np.sum).replace(np.nan,0)
temp2=train[['user_id_x','song_lang','score']]
scoretable2=temp2.groupby(['user_id_x','song_lang']).sum().reset_index()
temp3=train[['user_id_x','song_type','score']]
scoretable3=temp3.groupby(['user_id_x','song_type']).sum().reset_index()
traindata2=scoretable2.pivot_table(values='score',index='user_id_x',columns='song_lang',aggfunc=np.sum).replace(np.nan,0)
traindata3=scoretable3.pivot_table(values='score',index='user_id_x',columns='song_type',aggfunc=np.sum).replace(np.nan,0)
ff=pd.concat([traindata1,traindata2,traindata3],axis=1)


# In[248]:

## find user-user similarity for user to user-work feature matrix
from sklearn.metrics.pairwise import pairwise_distances
user_similarity_2 = pairwise_distances(ff, metric='cosine')
#item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')


# In[245]:


user_similarity_2


# In[250]:

## make prediction
user_prediction_2 = predict(train_data_matrix, user_similarity_2, type='user')


# In[255]:

## find top 10 user works for each user
prediction_user_2=pd.DataFrame({n: user_prediction_2.T[col].nlargest(10).index.tolist() 
                  for n, col in enumerate(item_prediction.T)}).T
user2=pd.DataFrame()
for i in range(100):
    rec=temp[temp['user_work_id'].isin(prediction_user_2.iloc[i])][['user_charact_y','song_lang','song_type']].drop_duplicates().apply(tuple, axis=1).reset_index().drop(['index'],axis=1).T
    user2=user2.append(rec)


# In[256]:

## test method 1 
testact=test.groupby('user_id_x').apply(lambda x:x[['user_charact_y','song_lang','song_type']].apply(tuple,axis=1)).reset_index()

num_user_method2=[testact[testact['user_id_x']==i][0].isin(user2.iloc[i]).sum() for i in range(100)]


# In[257]:

## test method 2
testact2=test.groupby('user_id_x').apply(lambda x:x[['user_id_x','user_charact_y','song_lang','song_type']])
num_user2_method2=[]
for i in range(100):
    rec=temp[temp['user_work_id'].isin(prediction_user_2.iloc[i])][['user_charact_y','song_lang','song_type']].drop_duplicates().mode(axis=0)
    num_user2_method2.append(testact2[testact2['user_id_x']==i][['user_charact_y','song_lang','song_type']].isin(rec.to_dict('list')).sum(axis=1).sum())


# In[258]:


print sum(num_user_method2),sum(num_user2_method2)


# ## Conclusion
# # based on random data method 2 is slightly better

# In[8]:

