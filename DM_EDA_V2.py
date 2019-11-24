#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 09:47:51 2019

@author: yuanyuan
"""


#%%-----------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
#import os
#os.chdir("/Users/yuanyuan/Desktop/DATS_6103/project/DM_data sets")


#%%-----------------------------------------------------------------------
# open dataset and view structure of each data set
aisles = pd.read_csv('aisles.csv')
departments = pd.read_csv('departments.csv')
train = pd.read_csv('order_products__train.csv')
prior = pd.read_csv('order_products__prior.csv')
orders = pd.read_csv('orders.csv')
products = pd.read_csv('products.csv')
sample = pd.read_csv('sample_submission.csv')

pd.set_option('display.max_columns',10)
all_sets = [aisles,departments,train,prior,orders,products,sample]
for i in all_sets:
    print(i.head())
    
    
#%%-----------------------------------------------------------------------
# Fisrt part: orders.csv
# 1.1 basic information about Orders dataset
    
print(orders.shape)
print(orders.columns)
print(orders.dtypes)
orders.count()


# 'days_since_prior_order': missing value
print(orders['days_since_prior_order'].dtype)
orders['days_since_prior_order'].replace(np.nan,0.0)

#%%-----------------------------------------------------------------------

# 1.2 Three sets
color = sns.color_palette()
dist_eval_set=orders.eval_set.value_counts()
sns.barplot(dist_eval_set.index,dist_eval_set.values, alpha=0.8,color=color[1])
plt.ylabel('orders')
plt.xlabel('eval_set type')
plt.title('Number of orders in each set')
plt.show()

dist_eval_set=orders.eval_set.value_counts()
print(dist_eval_set)
# There are 3214,784 orders for the prior set, and the dataset extract the last order of each custormer as train and test dataset, respectively.
# The train set has 131,209 observations and the test dataset has 75,000 observations.

    

color = sns.color_palette()
group_ev=orders.groupby('eval_set')
x=[]
y=[]
for name,group in group_ev:
    x.append(name)
    y.append(group.user_id.unique().shape[0])
sns.barplot(x,y, alpha=0.8,color=color[2])
plt.ylabel('users')
plt.xlabel('eval_set type')
plt.title('Number of users in each set')
plt.show()


group_ev=orders.groupby('eval_set')

for name,group in group_ev:
    print(name)
    print(group.user_id.unique().shape[0])

# There are 206209 users in prior set, 75000 users in test set and 131209 users in train set.
# There are 206,209 customers in total. Out of which, the last purchase of 131,209 customers are given as train set and we need to predict for the rest 75,000 customers.
#%%-----------------------------------------------------------------------
# 1.3 Order information
# 1.3.1 Frequency of orders (频数)
dist_no_orders=orders.groupby('user_id').order_number.max()
dist_no_orders=dist_no_orders.value_counts()

plt.figure(figsize=(20,8))
sns.barplot(dist_no_orders.index,dist_no_orders.values)
plt.xlabel('orders')
plt.ylabel('users')
plt.title('Frequency of orders by users')
plt.show()

#用boxplot验证是否有outliners
sns.boxplot(dist_no_orders.index)
plt.show()
# So there are no orders less than 4 and is max capped at 100 as given in the data page.

#%%-----------------------------------------------------------------------
# 1.3.2 Times of orders(热门购物时间)
# By day of week
dist_d_orders=orders.order_dow.value_counts()

sns.barplot(dist_d_orders.index,dist_d_orders.values, palette=sns.color_palette('Blues_d',7))
plt.xlabel('day of week')
plt.ylabel('orders')
plt.title('Frequency of orders by day of week')
plt.show()
# It looks as though 0 represents Saturday and 1 represents Sunday. Wednesday is then the least popular day to make orders.


#  By hour of day
dist_h_orders=orders.order_hour_of_day.value_counts()
plt.figure(figsize=(10,8))
sns.barplot(dist_h_orders.index,dist_h_orders.values, palette=sns.color_palette('Greens_d',24))
plt.xlabel('hour of day')
plt.ylabel('orders')
plt.title('Frequency of orders by hour of day')
plt.show()
# So majority of the orders are made during day time. The 10am hour is the most popular time to make orders, followed by a dip around lunch time and a pickup in the afternoon.Now let us combine the day of week and hour of day to see the distribution.


# By hour in a day
grouped=orders.groupby(['order_dow','order_hour_of_day']).order_number.count().reset_index() 
# using reset_index to set order_dow and ordr_h_day as columns, or they would be index
time_orders=grouped.pivot('order_dow', 'order_hour_of_day','order_number')

plt.figure(figsize=(10,5))
sns.heatmap(time_orders, cmap='YlOrRd')
plt.ylabel('Day of Week')
plt.xlabel('Hour of Day')
plt.title('Number of Orders Day of Week vs Hour of Day')
plt.show()
# Saturday afternoon and Sunday morning are the most popular time to make orders.


#%%-----------------------------------------------------------------------
# 1.3.3 Time interval(频率)
dist_d_prior_orders=orders.days_since_prior_order.value_counts()

plt.figure(figsize=(12,8))
sns.barplot(dist_d_prior_orders.index,dist_d_prior_orders.values, palette=sns.color_palette('Greens_d',31))
plt.xlabel('days of prior order')
plt.ylabel('count')
plt.title('Time interval between orders')
plt.show()

# While the most popular relative time between orders is monthly (30 days), there are "local maxima" at weekly (7 days), biweekly (14 days), triweekly (21 days), and quadriweekly (28 days).
# Looks like customers order once in every week (check the peak at 7 days) or once in a month (peak at 30 days). We could also see smaller peaks at 14, 21 and 28 days (weekly intervals).
#%%-----------------------------------------------------------------------









#%%-----------------------------------------------------------------------
# Second part: product.csv (merge products.csv, aisles.csv and departents.csv)
product = products.merge(aisles).merge(departments)
print(product.shape)
print(product.columns)
print(product.dtypes)
product.count()

#%%-----------------------------------------------------------------------
# 2.1 products in departments and aisles
# 2.1.1 products in departments
grouped = product.groupby("department")["product_id"].aggregate({'Total_products': 'count'}).reset_index()
grouped  = grouped.groupby(['department']).sum()['Total_products'].sort_values(ascending=False)
print(grouped)


plt.xticks(rotation='vertical')
sns.barplot(grouped.index, grouped.values)
plt.ylabel('Number of products', fontsize=13)
plt.xlabel('Departments', fontsize=13)
plt.title("The number of products in each department")
plt.show()

# The most five important departments are personal care, snacks, pantry, beverages and frozen. The number of items from these departments were more than 4,000 times.



# 2.1.2 products in asiles among all departments
grouped2 = product.groupby("aisle")["product_id"].aggregate({'Total_products': 'count'}).reset_index()
grouped2 = grouped2.sort_values(by='Total_products', ascending=False)[:20]
print(grouped2)

grouped2  = grouped2.groupby(['aisle']).sum()['Total_products'].sort_values(ascending=False)

plt.xticks(rotation='vertical')
sns.barplot(grouped2.index, grouped2.values)
plt.ylabel('Number of products', fontsize=13)
plt.xlabel('Aisles', fontsize=13)
plt.title("The number of products in each aisle(top 20)")
plt.show()
# The most three important aisles are canding chocolate, ice cream and vitamins sumpplements.



# 2.1.3 products in asiles among each department
grouped3 = product.groupby(['department','aisle'])
grouped3 = grouped3['product_id'].aggregate({'Total_products': 'count'}).reset_index()
print(grouped3)

fig, axes = plt.subplots(7,3, figsize=(20,45),gridspec_kw =  dict(hspace=1.4))
for (aisle, group), ax in zip(grouped3.groupby(["department"]), axes.flatten()):
    g = sns.barplot(group.aisle, group.Total_products , ax=ax)
    ax.set(xlabel = "Aisles", ylabel=" Number of products")
    g.set_xticklabels(labels = group.aisle,rotation=90, fontsize=12)
    ax.set_title(aisle, fontsize=15)
# 疑问：要做这么细致吗？
# Each graph shows the number of products in each aisle of different departments.  
    
    
    
    
    
    
#%%-----------------------------------------------------------------------
# 2.3 Oreder of product(merge products+all_order(prior+train)+orders) 
all_order = pd.concat([train, prior], axis=0)
order_flow = orders[['user_id', 'order_id']].merge(all_order[['order_id', 'product_id']]).merge(product)
order_flow.head()

print(order_flow.shape)
print(order_flow.columns)
print(order_flow.dtypes)
order_flow.count()


#%%-----------------------------------------------------------------------
# 2.3.1 Sales in each department(find best selling apartment)
grouped4 = order_flow.groupby("department")["order_id"].aggregate({'Total_orders': 'count'}).reset_index()
grouped4.sort_values(by='Total_orders', ascending=False, inplace=True)
print(grouped4)

grouped4  = grouped4.groupby(['department']).sum()['Total_orders'].sort_values(ascending=False)
plt.xticks(rotation='vertical')
sns.barplot(grouped4.index, grouped4.values)
plt.ylabel('Number of Orders', fontsize=13)
plt.xlabel('Departments', fontsize=13)
plt.title('Sales in each department')
plt.show()
# the most three popular departments are produce, dairy eggs and snacks.
#%%-----------------------------------------------------------------------
# 2.3.2 Sales in each aisle(best selling aisle)
grouped5 = order_flow.groupby("aisle")["order_id"].aggregate({'Total_orders': 'count'}).reset_index()
grouped5.sort_values(by='Total_orders', ascending=False, inplace=True )
print(grouped5.head(15))

grouped5 = grouped5.groupby(['aisle']).sum()['Total_orders'].sort_values(ascending=False)[:15]
plt.xticks(rotation='vertical')
sns.barplot(grouped5.index, grouped5.values)
plt.ylabel('Number of Orders', fontsize=13)
plt.xlabel('Aisles', fontsize=13)
plt.title('Sales in each aisle')
plt.show()

# the top three best selling aisles are fresh fruits, fresh vegetables and packaged vegetables fruits.
#%%-----------------------------------------------------------------------







#%%----------------------------------------------------------------------- 
# 2.4 Reorder of product
# 2.4.1 the number of reordered products
reord=sum(prior['reordered']==1)
not_reord=sum(prior['reordered']==0)
order_sum = reord + not_reord
reord_pro=reord/order_sum
not_ord_pro=not_reord/order_sum

all_order = pd.concat([train, prior], axis=0)
grouped6 = all_order.groupby("reordered")["product_id"].aggregate({'Total_products': 'count'}).reset_index()
grouped6 = grouped6.groupby(['reordered']).sum()['Total_products'].sort_values(ascending=False)
print(grouped6)
sns.barplot(grouped6.index,grouped6.values)
plt.ylabel('Number of Products', fontsize=13)
plt.xlabel('Reordered or Not Reordered', fontsize=13)
plt.ticklabel_format(style='plain', axis='y')
plt.title("Not reorder vs Reorder")
plt.show()

# conclusion:
print(reord, "products are previously ordered by customers, reordered products take",round(reord_pro,2),"% of ordered products.")
print(not_reord, "products are not ordered by customers before, non-reordered products take",round(not_ord_pro,2),"% of ordered products.")
#%%-----------------------------------------------------------------------
# 2.4.2 highest reordered rate 
# reorder_sum: 被重复购买的商品的总数; total: 所有被购买过的商品的总和
grouped7 = all_order.groupby("product_id")["reordered"].aggregate({'reorder_sum': sum,'total': 'count'}).reset_index()
grouped7['reord_ratio']= grouped7['reorder_sum'] / grouped7['total']
grouped7 = pd.merge(grouped7, product, how='left', on=['product_id'])
grouped8 = grouped7.sort_values(['reord_ratio'],ascending=False).head(10)
print(grouped8)

sns.barplot(grouped8['product_name'],grouped8['reord_ratio'])
plt.ylim([0.85,0.95])
plt.xticks(rotation='vertical')
plt.title('Top 10 reordered rate')
plt.show()

# conclusion: 
# 1.The three products with the highest reordered rate are Raw Veggie Wrappers, Serenity Ultimate Extrema Overnight Pads and Orange Energy Shots.
#%%-----------------------------------------------------------------------
# 2.4.3 department with highest reorder ratio 
grouped9 = grouped7.sort_values(['reord_ratio'],ascending=False)
sns.lineplot(grouped7['department'],grouped7['reord_ratio'])
plt.xticks(rotation='vertical')
plt.title('Reordered ratio in each department')
plt.show()

# A: Personal care has lowest reorder ratio and dairy eggs have highest reorder ratio.
#%%-----------------------------------------------------------------------
# 2.4.4 Relationship between add_to_cart and reordered?

# add_to_cart_order: The sequence of product is added to the cart in each order

prior["add_to_cart_order_mod"] =prior["add_to_cart_order"].copy()
prior["add_to_cart_order_mod"].loc[prior["add_to_cart_order_mod"]>70] = 70
grouped_df = prior.groupby(["add_to_cart_order_mod"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(grouped_df['add_to_cart_order_mod'].values, grouped_df['reordered'].values, alpha=0.8)
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Add to cart order', fontsize=12)
plt.title("Add to cart order - Reorder ratio", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()

# t-test
data1 = prior[prior['reordered']==0]['add_to_cart_order']
data2 = prior[prior['reordered']==1]['add_to_cart_order']
print(np.mean(data1))
print(np.mean(data2))
stats.ttest_ind(data1,data2)

# conclusion: 
# 1.Orders placed initially in the cart are more likely to be reorderd than one placed later in the cart.
# 2.We did t-test to verify whether the sequence of adding to cart are siginificantly different between reordered products and not reordered products.
# We can conclude from the results showing the p-value is smaller than 0.05 that the sequence of adding to cart significantly influence whether the products being reordered.
#%%-----------------------------------------------------------------------
