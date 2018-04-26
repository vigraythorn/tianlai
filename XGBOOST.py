
# coding: utf-8

# In[1]:


##import graphlab
import pandas as pd
import numpy as np
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn import metrics


# In[2]:


temp=pd.read_csv(r"/Users/wagao/Desktop/Independent Study/Tianlai-project/tianlai-master/writeNewDataLog20180315.csv",sep='\t',encoding="utf-8",engine='python',)


# In[3]:


temp.columns


# In[4]:


from collections import Counter

Counter(temp.time)


# In[5]:


temp.loc[temp.activity==u'送礼物','score']=12
temp.loc[temp.activity==u'评论','score']=10
temp.loc[temp.activity==u'播放','score']=2
temp.loc[temp.activity==u'下载','score']=6

temp1=temp.groupby(['user_id_x','user_work_id','constellation_x','gender_x','region_x','user_age_x','user_age_y','region_y','gender_y','constellation_y','song_lang','song_genre','singer_m','singer_f','singer_g','singer_sex'],as_index=False)['score'].sum()
temp1.columns


# In[6]:


from collections import Counter
Counter(temp1.score)


# In[6]:


y=temp1[['score']]
X=temp1[['gender_x','constellation_x','region_x','user_age_x','user_age_y','region_y','gender_y','constellation_y','song_lang','song_genre','singer_m','singer_f','singer_g','singer_sex']]
 

X['gender_x'] = X['gender_x'].astype('category')
X['constellation_x'] = X['constellation_x'].astype('category')
X['region_x'] = X['region_x'].astype('category')
X['user_age_x'] = X['user_age_x'].astype('category')
X['gender_y'] = X['gender_y'].astype('category')
X['constellation_y'] = X['constellation_y'].astype('category')
X['region_y'] = X['region_y'].astype('category')
X['user_age_y'] = X['user_age_y'].astype('category')
X['song_lang'] = X['song_lang'].astype('category')
X['song_genre'] = X['song_genre'].astype('category')
X['singer_m'] = X['singer_m'].astype('category')
X['singer_f'] = X['singer_f'].astype('category')
X['singer_g'] = X['singer_g'].astype('category')
X['singer_sex'] = X['singer_sex'].astype('category')

cat_columns = X.select_dtypes(['category']).columns
X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)

X['gender_x'] = X['gender_x'].astype(str)
X['constellation_x'] = X['constellation_x'].astype(str)
X['region_x'] = X['region_x'].astype(str)
X['user_age_x'] = X['user_age_x'].astype(str)
X['gender_y'] = X['gender_y'].astype(str)
X['constellation_y'] = X['constellation_y'].astype(str)
X['region_y'] = X['region_y'].astype(str)
X['user_age_y'] = X['user_age_y'].astype(str)
X['song_lang'] = X['song_lang'].astype(str)
X['song_genre'] = X['song_genre'].astype(str)
X['singer_m'] = X['singer_m'].astype(str)
X['singer_f'] = X['singer_f'].astype(str)
X['singer_g'] = X['singer_g'].astype(str)
X['singer_sex'] = X['singer_sex'].astype(str)


offset=int(temp1.shape[0]*0.8)

X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

y_test.sort_values(by='score', ascending=False, na_position='first')


# In[7]:


from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import math

params = {'n_estimators': 1000, 'max_depth': 100, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'huber','min_samples_leaf': 50, 'max_features': 'sqrt','subsample': 0.8}
clf1 = ensemble.GradientBoostingRegressor(**params)

clf1.fit(X_train, y_train)
#fpr, tpr, thresholds = metrics.roc_curve(y_test, clf.predict(X_test), pos_label=2)
#metrics.auc(fpr, tpr)
mse = mean_squared_error(y_test, clf1.predict(X_test))
#mse2 = mean_squared_error(y_train, clf1.predict(X_train))
mse=math.sqrt( mse )

print("MSE: %.4f" % mse)
#print("MSE2: %.4f" % mse2)


 


# In[13]:


import matplotlib.pyplot as plt

feature_importance = clf1.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# In[8]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import math

params = {'n_estimators': 1000, 'max_depth': 1000,'random_state': 2, 'random_state': 2 }
clf2 =RandomForestRegressor(**params)

clf2.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf2.predict(X_test))
mse=math.sqrt( mse )

print("MSE: %.4f" % mse)


# In[13]:


import matplotlib.pyplot as plt

feature_importance = clf2.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# In[9]:


from sklearn import svm
from sklearn.datasets import make_regression
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import math
   
params = {'C': 2.0, 'cache_size': 300,'coef0': 0.0, 'degree': 5 ,'epsilon': 0.9, 'gamma': 'auto','kernel': 'rbf', 'max_iter': -1,'shrinking': True,'tol': 0.01,'verbose': False}
clf3 =svm.SVR(**params)

clf3.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf3.predict(X_test))
mse=math.sqrt( mse )

print("MSE: %.4f" % mse)


# In[15]:


import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 100, 100

lw = 1
plt.scatter(X_test.index, y_test, color='darkorange', label='data')
#plt.plot(X_test.iloc[:,0], clf1.predict(X_test), color='navy', lw=lw, label='BOOST model')
plt.plot(X_test.index, clf1.predict(X_test), color='navy', lw=lw, label='BOOST model')
plt.plot(X_test.index, clf2.predict(X_test), color='yellow', lw=lw, label='RANDOM model')
plt.plot(X_test.index, clf3.predict(X_test), color='red', lw=lw, label='SVM model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('RECOMMENDATION ENGINE')
plt.legend()
plt.show()

