#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


orders=pd.read_csv('orders.csv')
order_products_prior=pd.read_csv('order_products__prior.csv')
department=pd.read_csv('departments.csv')
aisle=pd.read_csv('aisles.csv')
product=pd.read_csv('products.csv')
order_products_train=pd.read_csv('order_products__train.csv')


# # preprocessing

# ## check order_id for train/prior set and orders set

# In[3]:


# order_id for train
print(order_products_train.order_id.value_counts().shape[0])
print(orders.loc[orders.eval_set=='train'].shape[0])
# order_id for prior
print(order_products_prior.order_id.value_counts().shape[0])
print(orders.loc[orders.eval_set=='prior'].shape[0])


# From above, we can find that the order_id for both train & prior dataset and orders set is correspondont one by one, which is consistent with data description in the kaggle website overview.

# ## missing value

# In[4]:


print(orders.isnull().sum())
orders.index


# There are 206209 missing values in days_since_piror_orders, they means these orders are first order for the user, respectively. According to this meaning, we replace NaN with 0.

# In[5]:


orders=orders.fillna(0)
orders.isnull().sum()


# ## subsetting and merging for datasets

# In[6]:


#merge order_products_prior and orders 
merge_oop=pd.merge(orders, order_products_prior, on='order_id',how='right') #equals to select evalset==0
merge_oop.head()


# In[7]:


#check merge results
print(order_products_prior.loc[order_products_prior['order_id']==2539329])
print(merge_oop.loc[merge_oop['order_id']==2539329])


# As for training, all of the data we used to analyze should come from prior relevant information excluding the last order information for each user, for the next feature selection, we only use 'merge_oop' we created just now instead of orders and order_products_prior. 

# Smillarly, 'order' dataset includes prior orders and last order used for training label. Therefore, we need to exclude information about last order for each user before we using this dataset.

# In[8]:


orders_prior=orders.loc[orders.eval_set=='prior']
orders_prior.head()


# # 2.feature selection

# ## 1.user feature

# ### 1.1 number of orders for each user

# In[9]:


user_f_1=merge_oop.groupby('user_id').order_number.max().reset_index(name='n_orders_users')
user_f_1.head()


# ### 1.2 number of products for each user

# In[10]:


user_f_2=merge_oop.groupby('user_id').product_id.count().reset_index(name='n_products_users')
user_f_2.head()


# ### 1.3 average products of order for each user 

# In[11]:


user_f_2['avg_products_users']=user_f_2.n_products_users/user_f_1.n_orders_users
user_f_2.head()


# ### 1.4 day of one week orderd most for each user

# In[12]:


temp=merge_oop.groupby('user_id')['order_dow'].value_counts().reset_index(name='times_d')
temp.head()


# In[13]:


# find most days ordered by each user
user_f_3=temp.loc[temp.groupby('user_id')['times_d'].idxmax(),['user_id','order_dow']]
user_f_3=user_f_3.rename(columns={'order_dow':'dow_most_user'})   
user_f_3.head()
         


# ### 1.5 time of one day ordered most for each user

# In[14]:


temp=merge_oop.groupby('user_id')['order_hour_of_day'].value_counts().reset_index(name='times_h')
temp.head()


# In[15]:


# find most hours in one day ordered by each user
user_f_4=temp.loc[temp.groupby('user_id')['times_h'].idxmax(),['user_id','order_hour_of_day']]
user_f_4=user_f_4.rename(columns={'order_hour_of_day':'hod_most_user'})   
user_f_4.head()


# ### 1.6 reorder_ratio for each user

# In[16]:


user_f_5=merge_oop.groupby('user_id').reordered.mean().reset_index(name='reorder_ratio_user')
user_f_5.head()


# ### 1.7 shopping frequency for each user

# In[17]:


order_user_group=orders_prior.groupby('user_id')
user_f_6=(order_user_group.days_since_prior_order.sum()/order_user_group.days_since_prior_order.count()).reset_index(name='shopping_freq')
user_f_6.head()


# ### user feature list

# In[18]:


user=pd.merge(user_f_1,user_f_2,on='user_id')
user=user.merge(user_f_3,on='user_id')
user=user.merge(user_f_4,on='user_id')
user=user.merge(user_f_5,on='user_id')
user=user.merge(user_f_6,on='user_id')
user.head()


# ## 2.product feature

# ### 2.1 times of ordered for each product

# In[19]:


prod_f_1=order_products_prior.groupby('product_id').order_id.count().reset_index(name='times_bought_prod')
prod_f_1.head()


# ### 2.2 reordered ratio for each product

# In[20]:


prod_f_2=order_products_prior.groupby('product_id').reordered.mean().reset_index(name='reorder_ratio_prod')
prod_f_2.head()


# ### 2.3 average postions of cart for each product

# In[21]:


prod_f_3=order_products_prior.groupby('product_id').add_to_cart_order.mean().reset_index(name='position_cart_prod')
prod_f_3.head()


# ### 2.4 reordered ratio for each department

# In[22]:


prod_dep=pd.merge(department['department_id'],product[['department_id','product_id']],on='department_id',how='right')
totall_info=pd.merge(prod_dep,prod_f_2,on='product_id',how='right')
totall_info.head()


# In[23]:


group=totall_info.groupby('department_id')
prod_f_4=group.reorder_ratio_prod.mean().reset_index(name='reorder_ratio_dept')

prod_f_4=pd.merge(prod_f_4,totall_info,on='department_id')
del prod_f_4['reorder_ratio_prod']
prod_f_4.drop(['department_id'],axis=1,inplace=True)
prod_f_4.head()


# ### product feature list

# In[24]:


prod=pd.merge(prod_f_1,prod_f_2,on='product_id')
prod=prod.merge(prod_f_3,on='product_id')
prod=prod.merge(prod_f_4,on='product_id')
prod.head()


# ## 3. user & product feature

# ### 3.1 times of one product bought by one user

# In[25]:


user_prd_f_1=merge_oop.groupby(['user_id','product_id']).order_id.count().reset_index(name='times_bought_up')
user_prd_f_1.head()


# ### 3.2 reordered ratio of one product bought by one user

# In[26]:


# number of orders for one user
user_prd_f_2=merge_oop.groupby('user_id').order_number.max().reset_index(name='n_orders_users')
# when the user bought the product for the first time 
temp=merge_oop.groupby(['user_id','product_id']).order_number.min().reset_index(name='first_bought_number')
# merge two datasets
user_prd_f_2=pd.merge(user_prd_f_2,temp,on='user_id')
# how many orders performed after the user bought the product for the first time
user_prd_f_2['order_range']=user_prd_f_2['n_orders_users']-user_prd_f_2['first_bought_number']+1
#reordered ratio 
user_prd_f_2['reorder_ratio_up']=user_prd_f_1.times_bought_up/user_prd_f_2.order_range
user_prd_f_2.head()


# In[27]:


user_prd_f_2=user_prd_f_2.loc[:,['user_id','product_id','reorder_ratio_up']]
user_prd_f_2.head()


# ### 3.3 ratio of one product bought in one user's last four orders

# In[28]:


#Reversing the order number for each product.
merge_oop['order_number_back']=merge_oop.groupby('user_id')['order_number'].transform(max)-merge_oop['order_number']+1
merge_oop.head()


# In[29]:


# select orders where order_number_back <=4
temp1=merge_oop.loc[merge_oop['order_number_back']<=4]
temp1.head()


# In[30]:


# create feature
user_prd_f_3=(temp1.groupby(['user_id','product_id'])['order_number_back'].count()/4).reset_index(name='ratio_last4_orders_up')
user_prd_f_3.head()


# ### user & product feature list

# In[31]:


# merge three features for the user&product list
user_prd=pd.merge(user_prd_f_1,user_prd_f_2,on=['user_id','product_id'])
user_prd=user_prd.merge(user_prd_f_3,on=['user_id','product_id'],how='left')
user_prd.head()


# After checking, we notice that some rows for raio_last4_orders_up have NaN values for our new feature. This happens as there might be products that the customer did not buy on its last four orders. For these cases, we turn NaN values into 0.

# In[32]:


user_prd.ratio_last4_orders_up.fillna(0,inplace=True)


# ## total features list

# In[33]:


total_info=pd.merge(user_prd, user,on='user_id',how='left')
total_info=total_info.merge(prod,on='product_id',how='left')
total_info.head()


# In[34]:


total_info.info()


# In[35]:


total_info.head()


# Totally, we select 14 features to do our models futher. They are'n_orders_users', 'n_products_users', 'avg_products_users',
#        'dow_most_user','hod_most_user','reorder_ratio_user', 'shopping_freq',
#        'product_id', 'times_bought_up', 'reorder_ratio_up',
#        'times_last4_orders_up', 'times_bought_prod', 'reorder_ratio_prod',
#        'position_cart_prod', 'reorder_ratio_dept'.

# In[36]:


total_info.isnull().sum()


# After checking, there is no missing value in this dataset. Then we will use it as observations to build our model.

# # 3.creating train and test dataset

# ### set reordered as independent columns

# In[37]:


#select order_id for train by orders.csv
orders_y=orders.loc[((orders.eval_set=='train') | (orders.eval_set=='test')),['user_id','order_id','eval_set']]
orders_y.head()


# In[38]:


# add order_id to the total_info
total_info=pd.merge(total_info,orders_y,on='user_id',how='left')
total_info.head()


# ## create train dataset

# In[39]:


# select reordered in order_products_train as our dependent variable
total_info_train=total_info[total_info.eval_set=='train']
total_info_train=pd.merge(total_info_train,order_products_train[['product_id','order_id','reordered']],on=['order_id','product_id'],how='left')
total_info_train.head()
#total_info=total_info.drop(['reordered_x','reordered_y','add_to_cart_order'],axis=1)


# We add 'reordered' to total_info, and this column in original order_products_train.csv means for the train order(last order for each user), if the product has been bought before.After merged, it means if the product of certain user has been bought before. dataset and the same t. Therefore, only reorder==1 columns have been selected and reorder==Nan means the product has not been selected in the user last order.

# In[40]:


# fill Nan with 0
total_info_train['reordered'].fillna(0,inplace=True) ## inplace decides whether modify original dataset


# In[41]:


# drop order_id
total_info_train.drop(['order_id','eval_set'],axis=1,inplace=True)
total_info_train.head()


# In[42]:


# reset 'user_id' and 'product_id'index 
total_info_train=total_info_train.set_index(['user_id','product_id'])
total_info_train.head()


# ### create test dataset

# In[43]:


# select test part
total_info_test=total_info[total_info.eval_set=='test']
total_info_test.head()


# In[44]:


# reset 'user_id' and 'product_id'index 
total_info_test=total_info_test.set_index(['user_id','product_id'])
total_info_test.head()


# In[45]:


# only select features
total_info_test.drop(['order_id','eval_set'],axis=1,inplace=True)
total_info_test.head()


# In[46]:


total_info_train.shape,total_info_test.shape


# ## 4.model building

# ## 4.1logistic regression

# In[47]:


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


# In[48]:


X=total_info_train.drop('reordered',axis=1)
y=total_info_train.reordered
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,stratify=y)


# In[49]:


y.value_counts()


# In[50]:


sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# In[51]:


clf = LogisticRegression()
clf.fit(X_train, y_train)


# In[52]:


clf.coef_


# In[53]:


# make predictions# predicton on test
y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)


# In[54]:


print("\n")

print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")


print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)
print("\n")


# In[66]:


conf_matrix = confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(conf_matrix, index=['0','1'], columns=['0','1'] )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()


# ## 4.2Naive Bayes

# In[56]:


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[57]:


X=total_info_train.drop('reordered',axis=1)
y=total_info_train.reordered


# In[58]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[59]:


clf = GaussianNB()
clf.fit(X_train, y_train)


# In[60]:


y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)


# In[61]:


print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")

print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)
print("\n")


# In[64]:


conf_matrix = confusion_matrix(y_test, y_pred)
#class_names = total_info_train['reordered'].unique()


df_cm = pd.DataFrame(conf_matrix, index=['0','1'], columns=['0','1'] )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()
plt.show()

# In[65]:
# # Decision Tree

# In[108]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt


# In[130]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


# In[131]:


clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)


# In[132]:


y_pred_gini = clf_gini.predict(X_test)


# In[133]:


print("Results Using Gini Index: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_gini))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_gini) * 100)


# In[134]:


conf_matrix = confusion_matrix(y_test, y_pred_gini)
df_cm = pd.DataFrame(conf_matrix)

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()


# # Random Forest

# In[135]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)


# In[93]:


# # plot feature importances
# # get feature importances
# importances = clf.feature_importances_

# # convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
# f_importances = pd.Series(importances, data_train.columns)

# # sort the array in descending order of the importances
# f_importances.sort_values(ascending=False, inplace=True)

# # make the bar Plot from f_importances
# f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=15)

# # show the plot
# plt.tight_layout()
# plt.show()


# In[136]:


# predicton on test using all features
y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)


# In[137]:


from sklearn.metrics import roc_auc_score
print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")
print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)


# In[138]:


conf_matrix = confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(conf_matrix)

plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)

hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()

# # KNN

# In[143]:


# from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)


# In[144]:


y_pred = clf.predict(X_test)


# In[145]:


print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")


# In[146]:


conf_matrix = confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(conf_matrix)

plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)

hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()
