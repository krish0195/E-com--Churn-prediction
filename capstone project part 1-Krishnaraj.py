#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[10]:


df=pd.read_csv(r'C:\Users\Dell\Desktop\GREAT LAKES\back to studies\capstone\Customer+Churn+data.csv')


# In[11]:


df


# In[12]:


df.head()


# In[13]:


#Feature Removal
df=df.drop(["AccountID"],axis=1)


# In[14]:


df.shape


# In[15]:


df.describe(include="all").T


# In[16]:


df.info()


# In[17]:


df.duplicated().sum()


# In[18]:


#Feature replacement
df["Gender"] = df["Gender"].replace("M", 'Male')
df["Gender"] = df["Gender"].replace("F", 'Female')

df["account_segment"] = df["account_segment"].replace("Regular +","Regular Plus")
df["account_segment"] = df["account_segment"].replace("Super +","Super Plus")


# In[19]:


len(df)


# In[20]:


#Check Data Balancing
df["Churn"] = df["Churn"].astype("int")
df['Churn'].value_counts(1)


# In[21]:



xx=df['Churn'].value_counts(1).reset_index()
plt.figure(figsize=(10,5))
sns.barplot(x="index",y="Churn",data=xx,palette="cividis");
plt.show()


# In[22]:


df.isna().sum().sort_values(ascending=False)/len(df)


# missing vale 
# mv>2% <10 % impute by mode

# In[23]:


# Replacing the merical values and missing value treatment

df["Tenure"] = df["Tenure"].replace("#",0)
df["Tenure"]=df["Tenure"].replace(np.nan,0)
df=df.astype({"Tenure":"int"})
#df["Tenure"] = df["Tenure"].astype(str).astype(int)
#df['Tenure'] = pd.to_numeric(df['Tenure'])
#df['Tenure'].astype(float).astype('Int64')
#df = df.astype({"Tenure": int})

df['Tenure'].replace(regex=True, inplace=True, to_replace=r'[^0-9.\-]',
value=r'') 
df['Tenure'] = df['Tenure'].astype(float)


df["CC_Contacted_LY"]=df["CC_Contacted_LY"].replace(np.nan,0)
df=df.astype({"CC_Contacted_LY":"int"})
#df['CC_Contacted_LY'] = pd.to_numeric(df['CC_Contacted_LY'])

df["rev_per_month"]=df["rev_per_month"].replace("+",0)
df["rev_per_month"]=df["rev_per_month"].replace(np.nan,0)
df=df.astype({"rev_per_month":"int"})
#df['rev_per_month'] = pd.to_numeric(df['rev_per_month'])

df["rev_growth_yoy"]=df["rev_growth_yoy"].replace("$",0)
df[" rev_growth_yoy"]=df["rev_growth_yoy"].replace(np.nan,0)
df=df.astype({"rev_growth_yoy":"int"})
#df=df.astype({"rev_growth_yoy":"int"})

df['rev_growth_yoy'] = pd.to_numeric(df['rev_growth_yoy'])

df["coupon_used_for_payment"]=df["coupon_used_for_payment"].replace("#",0)
df["coupon_used_for_payment"]=df["coupon_used_for_payment"].replace("$",0)
df["coupon_used_for_payment"]=df["coupon_used_for_payment"].replace("*",0)
df[" coupon_used_for_payment"]=df["coupon_used_for_payment"].replace(np.nan,0)

df["Day_Since_CC_connect"]=df["Day_Since_CC_connect"].replace("$",0)
#df['Day_Since_CC_connect'] = pd.to_numeric(df['Day_Since_CC_connect'])

df["cashback"]=df["cashback"].replace("$",0)


# In[24]:


df['Tenure'].fillna(df['Tenure'].median(), inplace=True)
df['Day_Since_CC_connect'].fillna(df['Day_Since_CC_connect'].median(), inplace=True)
df['rev_growth_yoy'].fillna(df['rev_growth_yoy'].median(), inplace=True)
df['rev_per_month'].fillna(df['rev_per_month'].median(), inplace=True)
df['CC_Contacted_LY'].fillna(df['CC_Contacted_LY'].median(), inplace=True)
df['rev_growth_yoy'].fillna(df['rev_growth_yoy'].median(), inplace=True)


# In[25]:


# Removing special char in catgorical value
df.Account_user_count=df.Account_user_count.replace(to_replace='@',value=df['Account_user_count'].mode()[0])
#df.Account_user_count=df.Account_user_count.replace(to_replace="",value=0)
df.Login_device=df.Login_device.replace(to_replace="&&&&",value=df['Login_device'].mode()[0])


# In[26]:


df['account_segment']=df['account_segment'].fillna(df['account_segment'].mode()[0])


# In[27]:


df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['account_segment'].fillna(df['account_segment'].mode()[0], inplace=True) 
df['Account_user_count'].fillna(df['Account_user_count'].mode()[0], inplace=True) 
df['Login_device'].fillna(df['Login_device'].mode()[0], inplace=True) 
df['City_Tier'].fillna(df['City_Tier'].mode()[0], inplace=True)
df['Service_Score'].fillna(df['Service_Score'].mode()[0],inplace=True)
df['CC_Agent_Score'].fillna(df['CC_Agent_Score'].mode()[0],inplace=True)
df['Marital_Status'].fillna(df['Marital_Status'].mode()[0],inplace=True)
df['Payment'].fillna(df['Payment'].mode()[0],inplace=True)
df['Complain_ly'].fillna(df['Complain_ly'].mode()[0],inplace=True)
df['Login_device'].fillna(df['Login_device'].mode()[0],inplace=True)
df['cashback'].fillna(df['cashback'].mode()[0],inplace=True)


# cateogry_columns=df.select_dtypes(include=['object']).columns.tolist()
# integer_columns=df.select_dtypes(include=['int64','float64']).columns.tolist()
# 
# for column in df:
#     if df[column].isnull().any():
#         if(column in cateogry_columns):
#             df[column]=df[column].fillna(df[column].mode()[0])
#         else:
#             df[column]=df[column].fillna(df[column].mean)

# In[28]:


missing_count_df = df.isnull().sum()
value_count_df = df.isnull().count()
missing_percentage_df = round(missing_count_df/value_count_df*100,2)


# In[29]:


plt.figure(figsize=(10,5))
missing_df = pd.DataFrame({'count' : missing_count_df, 'percentage' : missing_percentage_df })
barchart = missing_df.plot.bar(y='percentage',rot=90,figsize=(10,5))
plt.show()


# In[30]:


#Outliner check Before IQR Treatment
f, ax = plt.subplots(figsize=(10,5))
ax = sns.boxplot(data=df,width=0.6,palette="Set3",orient='h',linewidth=0.5)


# In[31]:


#Univariate analysis-Tenure
sns.histplot(df["Tenure"])


# In[32]:


sns.distplot(df['Tenure']);


# Tenure_range = pd.cut(df["Tenure"], 
#                      bins=[0, 20, 40, 60, 80, 100, 120, 140], 
#                      labels=["0-20", "20-40", "40-60", "60-80", "80-100", "100-120", "120-140"])
# 
# df["Tenure_range"].value_counts().sort_index().plot(kind="bar")
# plt.title("Tenure in each Tenure Range")
# plt.show()

# In[33]:


df['Tenure'].describe(include="all").T


# In[34]:


import matplotlib.pyplot as plt

df.boxplot(column=['Tenure'], grid=False, color='black')


# In[35]:


sns.boxplot(x="CC_Contacted_LY",data=df);


# In[36]:


df['CC_Contacted_LY'].describe(include="all").T


# In[37]:


sns.histplot(df["account_segment"])


# In[38]:


df['rev_per_month'].describe(include="all").T


# In[39]:


sns.boxplot(x="rev_per_month",data=df);


# In[40]:


df['rev_growth_yoy'].describe(include="all").T


# In[41]:


df['Account_user_count'] = df['Account_user_count'].astype('float')
sns.boxplot(x="Account_user_count",data=df);


# In[42]:


df['coupon_used_for_payment'] = df['coupon_used_for_payment'].astype('float')
sns.boxplot(x="coupon_used_for_payment",data=df);


# In[43]:



df["coupon_used_for_payment"] = df["coupon_used_for_payment"].astype("int")
df['coupon_used_for_payment'].describe(include="all").T


# In[44]:


df['cashback'] = df['cashback'].astype('int')
sns.boxplot(x="cashback",data=df);


# In[45]:



df['cashback'].describe(include="all").T


# In[46]:


df['coupon_used_for_payment'] = df['coupon_used_for_payment'].astype('float')
sns.boxplot(x="coupon_used_for_payment",data=df);


# In[47]:



df["Day_Since_CC_connect"] = df["Day_Since_CC_connect"].astype("int")
sns.boxplot(x="Day_Since_CC_connect",data=df);


# In[48]:


df['Day_Since_CC_connect'].describe(include="all").T


# In[49]:


sns.countplot(x='coupon_used_for_payment', data=df)


# In[50]:


# BAR plot


# In[51]:


sns.countplot(x="Gender",data=df)


# In[52]:


df['cashback'].astype(float)
sns.barplot(x="Gender",y="cashback",data=df)
plt.show()


# In[53]:


sns.countplot(x="Payment",data=df)


# In[54]:


sns.countplot(x="City_Tier",data=df)


# In[55]:


sns.countplot(x="Service_Score",data=df)


# In[56]:


sns.countplot(x="account_segment",data=df)


# In[57]:


#Bivarate


# In[58]:


sns.barplot(x="Churn",y="Tenure",data =df)


# In[59]:


df=df.astype({"Churn":"int"})
plt.rcParams["figure.figsize"] = (20,10)
sns.catplot(x='Account_user_count', col='Churn',hue="Gender", kind='count', data=df);
plt.show()


# In[60]:


sns.catplot(x='City_Tier', col='Payment', kind='count', data=df);
plt.show()


# In[61]:


#sns.barplot(x=df.Account_user_count,y=df.account_segment, hue='Churn', data=df)
#sns.catplot(x='account_segment', col='Account_user_count',hue="Churn", kind='count', data=df);
#sns.catplot(x='account_segment', col='Churn',hue="Gender", kind='count', data=df);


# In[62]:


sns.catplot(x='account_segment', col='Churn',hue='Gender', kind='count', data=df);
plt.show()


# In[63]:


sns.catplot(x='Service_Score', col='Churn',hue='Gender', kind='count', data=df);
plt.show()


# In[64]:


sns.catplot(x='Marital_Status', col='Churn',hue='Gender', kind='count', data=df);
plt.show()


# In[65]:


sns.catplot(x='City_Tier', col='Login_device',hue='Churn', kind='count', data=df);
plt.show()


# In[66]:


sns.catplot(x='City_Tier', col='Payment',hue='Churn', kind='count', data=df);
plt.show()


# In[67]:


from scipy import stats
plt.figure(figsize=(10,5))
sns.scatterplot(x="Tenure", y="cashback", data=df, alpha=0.3)
plt.title("Correlation between cashback and tenure")
plt.show()


# In[68]:


plt.figure(figsize=(10,5))
df=df.astype({"Day_Since_CC_connect":"int"})
sns.scatterplot(x="Day_Since_CC_connect", y="cashback", data=df, alpha=0.3)
plt.show()


# In[ ]:





# In[69]:



sns.scatterplot(x="rev_growth_yoy", y="rev_per_month", data=df, alpha=0.3)
plt.show()


# In[70]:


# construct box plot for continuous variables
cont=df.dtypes[(df.dtypes!='uint8') & (df.dtypes!='object') & (df.dtypes!='bool')].index
plt.figure(figsize=(10,5))
df[cont].boxplot(vert=0)
plt.title('With Outliers',fontsize=16) 
plt.show()


# In[71]:



def remove_outlier(col):
    sorted(col)
    Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range


# In[72]:


for column in df[cont].columns:
    lr,ur=remove_outlier(df[column]) 
    df[column]=np.where(df[column]>ur,ur,df[column])
    df[column]=np.where(df[column]<lr,lr,df[column])


# In[73]:


plt.figure(figsize=(10,5))
df[cont].boxplot(vert=0)
plt.title('After Outlier Removal',fontsize=16)
plt.show()


# In[74]:



corr = df[cont].corr()
f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corr, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()


# In[75]:


dfCopy = df.copy()


# In[76]:


dfCopy.head()


# In[ ]:





# In[77]:


#Converting all objects to categorical codes:Label encoding
dfCopy ['Payment']=np.where(dfCopy ['Payment'] =='Cash on Delivery', '1', dfCopy ['Payment']) 
dfCopy ['Payment']=np.where(dfCopy ['Payment'] =='Credit Card', '2', dfCopy ['Payment']) 
dfCopy ['Payment']=np.where(dfCopy ['Payment'] =='Debit Card', '3', dfCopy ['Payment']) 
dfCopy ['Payment']=np.where(dfCopy ['Payment'] =='E wallet', '4', dfCopy ['Payment']) 
dfCopy ['Payment']=np.where(dfCopy ['Payment'] =='UPI', '5', dfCopy ['Payment']) 

dfCopy ['Gender']=np.where(dfCopy ['Gender'] =='Male', '1', dfCopy ['Gender']) 
dfCopy ['Gender']=np.where(dfCopy ['Gender'] =='Female', '2', dfCopy ['Gender'])

dfCopy ['account_segment']=np.where(dfCopy ['account_segment'] =='HNI', '1', dfCopy ['account_segment']) 
dfCopy ['account_segment']=np.where(dfCopy ['account_segment'] =='Regular', '2', dfCopy ['account_segment']) 
dfCopy ['account_segment']=np.where(dfCopy ['account_segment'] =='Regular Plus', '3', dfCopy ['account_segment']) 
dfCopy ['account_segment']=np.where(dfCopy ['account_segment'] =='Super', '4', dfCopy ['account_segment'])
dfCopy ['account_segment']=np.where(dfCopy ['account_segment'] =='Super Plus', '5', dfCopy ['account_segment']) 



dfCopy ['Marital_Status']=np.where(dfCopy ['Marital_Status'] =='Divorced', '1', dfCopy ['Marital_Status']) 
dfCopy ['Marital_Status']=np.where(dfCopy ['Marital_Status'] =='Married', '2', dfCopy ['Marital_Status']) 
dfCopy ['Marital_Status']=np.where(dfCopy ['Marital_Status'] =='Single', '3', dfCopy ['Marital_Status']) 


dfCopy ['Login_device']=np.where(dfCopy ['Login_device'] =='Computer', '1', dfCopy ['Login_device']) 
dfCopy ['Login_device']=np.where(dfCopy ['Login_device'] =='Mobile', '2', dfCopy ['Login_device']) 

dfCopy ['Login_device'] = dfCopy ['Login_device'].fillna(dfCopy ['Login_device'].mode()[0])


# In[78]:


dfCopy.head()


# In[79]:


df=df.drop("Churn", axis=1)


# In[80]:


#Perform the K-Means clustering
#Standardize the data
# dropping categorails values

df=df.drop(["Payment","Gender","account_segment","Marital_Status","Login_device","Service_Score","City_Tier","CC_Agent_Score","Complain_ly"],axis = 1)


# In[81]:


df.head()


# In[82]:


from sklearn.preprocessing import StandardScaler
X = StandardScaler()
scaled_df = X.fit_transform(df)
scaled_df


# In[83]:


scaled_df = pd.DataFrame(scaled_df, index=df.index, columns=df.columns)
scaled_df.head()


# In[84]:


#Find the Within Sum of Squares (WSS) for 2 to 15 clusters.
from sklearn.cluster import KMeans
wss =[]
for i in range(1,15):
    KM = KMeans(n_clusters=i,random_state=1)
    KM.fit(scaled_df)
    wss.append(KM.inertia_)


# In[85]:


#Plot the Within Sum of Squares (WSS) plot using the values of 'inertia


# In[86]:


plt.plot(range(1,15), wss)
plt.grid()
plt.show()


# In[87]:


#Ideal No of clusters is 3
k_means6 = KMeans(n_clusters = 3,random_state=1)
k_means6.fit(scaled_df)
labels_3 = k_means6.labels_
labels_3


# In[88]:


df['Kmeans_clusters'] = labels_3
df.head()


# In[89]:


df.to_csv("3 new no After clustering.csv")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




