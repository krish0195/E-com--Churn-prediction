#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Basic EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# Model Preparation
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


# Model Building
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
#
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

# Model Performance
from sklearn.metrics import roc_auc_score,roc_curve,classification_report,confusion_matrix

# Model Validation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE


# In[6]:


df=pd.read_csv(r'C:\Users\Dell\Desktop\GREAT LAKES\back to studies\capstone\Materials of part 1-EDA\newresult.csv')


# In[7]:


df


# In[8]:


#Feature Encoding


# In[9]:


df.info()


# In[10]:


df.Churn.value_counts(normalize=True)
#SMOTE


# In[11]:


df.shape


# In[12]:


X = df.drop("Churn", axis=1)
Y = df.pop("Churn")


# In[15]:


print('X',X.shape)
print('Y',Y.shape)


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)


# In[17]:


print('X_train',X_train.shape)
print('x_test',X_test.shape)


# In[18]:


print('y_train',Y_train.shape)
print('y_test',Y_test.shape)


# from imblearn.over_sampling import SMOTE
# sm = SMOTE(random_state = 42)
# X_train_oversampled, y_train_oversampled = sm.fit_sample(X_train, y_train)
# X_train = pd.DataFrame(X_train_oversampled, columns=X_train.columns)

# In[19]:


from imblearn.over_sampling import SMOTE
smote = SMOTE()
X, Y = smote.fit_resample(X, Y)
df = pd.concat([pd.DataFrame(X), pd.DataFrame(Y)], axis=1)


# In[20]:


print('X_train',X_train.shape)
print('x_test',X_test.shape)


# In[21]:


print('y_train',Y_train.shape)
print('y_test',Y_test.shape)


# # 1)Checking Multicollinearity & VIF

# In[22]:


#VIF Dataframe
vif_df = pd.DataFrame()
vif_df["feature"] = X_train.columns


# In[23]:


# Calculate VIF for each feature
vif_df["VIF"] = [variance_inflation_factor(X_train.values,i)
for i in range(len(X_train.columns))]
print(vif_df)


# In[24]:


X_train = X_train.drop(["Login_device","Service_Score","Account_user_count","cashback","rev_growth_yoy"], axis=1)
X_test = X_test.drop(["Login_device","Service_Score","Account_user_count","cashback","rev_growth_yoy"], axis=1)


# In[25]:


# Dropping and cecking the VIF Score


# In[26]:


#VIF Dataframe
vif_df = pd.DataFrame()
vif_df["feature"] = X_train.columns
#
# Calculate VIF for each feature
vif_df["VIF"] = [variance_inflation_factor(X_train.values,i)
                        for i in range(len(X_train.columns))]
print(vif_df)


# In[27]:


#Significant Features
print('Y_train',Y_train.shape)
print('Y_test',Y_test.shape)


# In[28]:



# Standard Scaler - Normality assuption
# MinMax - everywhere else

from sklearn.preprocessing import StandardScaler
std_scale = StandardScaler()
X_train = std_scale.fit_transform(X_train)
X_test = std_scale.transform(X_test)


# In[166]:


#Model Building


# In[167]:


#1. Decision Tree


# In[168]:


dtcl = DecisionTreeClassifier(random_state=1)
#dtcl.fit(X_train, Y_train)


# In[169]:


param_grid_dt = {
    'criterion': ['gini'],
    'max_depth': [10],
    'min_samples_leaf': [10], 
    'min_samples_split': [50],
}
gs_dtcl = GridSearchCV(dtcl, param_grid_dt, cv = 5, n_jobs = -1, verbose = 1)
gs_dtcl.fit(X_train, Y_train)


# In[170]:


#2. (a) Model Prediction


# In[171]:


gs_dtcl.fit(X_train, Y_train)
print(gs_dtcl.best_params_)
best_grid = gs_dtcl.best_estimator_
best_grid
#{'criterion': 'gini', 'max_depth': 12, 'min_samples_leaf': 50, 'min_samples_split': 450}


# In[172]:



ytrain_predict = best_grid.predict(X_train)
ytest_predict = best_grid.predict(X_test)


# In[173]:



ytest_predict
ytest_predict_prob=best_grid.predict_proba(X_test)
ytest_predict_prob
pd.DataFrame(ytest_predict_prob).head()


# In[174]:


#2. (b) Model Performance


# In[175]:


confusion_matrix(Y_train, ytrain_predict)


# In[176]:


confusion_matrix(Y_test, ytest_predict)


# In[177]:


cart_train_acc=best_grid.score(X_train,Y_train) 
cart_train_acc


# In[178]:


cart_test_acc=best_grid.score(X_test,Y_test)
cart_test_acc


# In[179]:


print(classification_report(Y_train, ytrain_predict))


# In[180]:


print(classification_report(Y_test, ytest_predict))


# In[181]:


#2. (c) ROC-AUC Graph


# In[182]:


# AUC and ROC for the training data

# predict probabilities
probs = gs_dtcl.predict_proba(X_train)

# keep probabilities for the positive outcome only
probs = probs[:, 1]

# calculate AUC
from sklearn.metrics import roc_auc_score
cart_train_auc = roc_auc_score(Y_train, probs)
print('AUC: %.3f' % cart_train_auc)

# calculate roc curve
from sklearn.metrics import roc_curve
cart_train_fpr, cart_train_tpr, cart_train_thresholds = roc_curve(Y_train, probs)
plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model
plt.plot(cart_train_fpr, cart_train_tpr, marker='.')

# show the plot
plt.show()


# In[183]:


# AUC and ROC for the test data

# predict probabilities
probs = gs_dtcl.predict_proba(X_test)

# keep probabilities for the positive outcome only
probs = probs[:, 1]

# calculate AUC
from sklearn.metrics import roc_auc_score
cart_test_auc = roc_auc_score(Y_test, probs)
print('AUC: %.3f' % cart_test_auc)

# calculate roc curve
from sklearn.metrics import roc_curve
cart_test_fpr, cart_test_tpr, cart_test_thresholds = roc_curve(Y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model
plt.plot(cart_test_fpr, cart_test_tpr, marker='.')

# show the plot
plt.show()


# In[184]:


#2. (d) Model Performance Metrices


# In[185]:


cart_metrics=classification_report(Y_train, ytrain_predict,output_dict=True)
df=pd.DataFrame(cart_metrics).transpose()
cart_train_f1=round(df.loc["1"][2],2)
cart_train_recall=round(df.loc["1"][1],2)
cart_train_precision=round(df.loc["1"][0],2)
print ('cart_train_precision ',cart_train_precision)
print ('cart_train_recall ',cart_train_recall)
print ('cart_train_f1 ',cart_train_f1)


# In[186]:


cart_metrics=classification_report(Y_test, ytest_predict,output_dict=True)
df=pd.DataFrame(cart_metrics).transpose()
cart_test_precision=round(df.loc["1"][0],2)
cart_test_recall=round(df.loc["1"][1],2)
cart_test_f1=round(df.loc["1"][2],2)
print ('cart_test_precision ',cart_test_precision)
print ('cart_test_recall ',cart_test_recall)
print ('cart_test_f1 ',cart_test_f1)


# In[187]:


#2. (e) Feature Importance


# In[188]:


#2. Logistic Regression


# In[189]:


model = LogisticRegression(C=1.0, 
                           class_weight=None, 
                           dual=False, 
                           fit_intercept=True,
                           intercept_scaling=1, 
                           l1_ratio=None, 
                           max_iter=100,
                           n_jobs=None, 
                           penalty='l2',
                           random_state=1, 
                           solver='liblinear', 
                           tol=0.0001, 
                           verbose=0,
                           warm_start=False)
model.fit(X_train, Y_train) 


# In[190]:


#1. (a) Model Prediction


# In[191]:


y_predict_train = model.predict(X_train)
log_train_acc = model.score(X_train, Y_train)
log_train_acc


# In[192]:


y_predict_test = model.predict(X_test)
log_test_acc = model.score(X_test, Y_test)
log_test_acc


# In[193]:


model.intercept_


# In[194]:


model.coef_


# In[195]:


#1. (b) Model Performance


# In[196]:


confusion_matrix(Y_train, y_predict_train)


# In[197]:


confusion_matrix(Y_test, y_predict_test)


# In[198]:


print(classification_report(Y_train, y_predict_train))


# In[199]:


print(classification_report(Y_test, y_predict_test))


# In[200]:


#1. (c) ROC-AUC Graph


# In[201]:


# AUC and ROC for the training data

# predict probabilities
probs = model.predict_proba(X_train)

# keep probabilities for the positive outcome only
probs = probs[:, 1]

# calculate AUC
from sklearn.metrics import roc_auc_score
log_train_auc = roc_auc_score(Y_train, probs)
print('AUC: %.3f' % log_train_auc)

# calculate roc curve
from sklearn.metrics import roc_curve
log_train_fpr, log_train_tpr, train_thresholds = roc_curve(Y_train, probs)
plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model
plt.plot(log_train_fpr, log_train_tpr, marker='.')

# show the plot
plt.show()


# In[202]:



# AUC and ROC for the test data

# predict probabilities
probs = model.predict_proba(X_test)

# keep probabilities for the positive outcome only
probs = probs[:, 1]

# calculate AUC
from sklearn.metrics import roc_auc_score
log_test_auc = roc_auc_score(Y_test, probs)
print('AUC: %.3f' % log_test_auc)

# calculate roc curve
from sklearn.metrics import roc_curve
log_test_fpr, log_test_tpr, test_thresholds = roc_curve(Y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model
plt.plot(log_test_fpr, log_test_tpr, marker='.')

# show the plot
plt.show()


# In[203]:


#1. (d) Model Performance Metrice


# In[204]:


log_metrics=classification_report(Y_train, y_predict_train,output_dict=True)
df=pd.DataFrame(log_metrics).transpose()
log_train_precision=round(df.loc["1"][0],2)
log_train_recall=round(df.loc["1"][1],2)
log_train_f1=round(df.loc["1"][2],2)
print ('log_train_precision ',log_train_precision)
print ('log_train_recall ',log_train_recall)
print ('log_train_f1 ',log_train_f1)


# In[205]:


log_metrics=classification_report(Y_test, y_predict_test,output_dict=True)
df=pd.DataFrame(log_metrics).transpose()
log_test_precision=round(df.loc["1"][0],2)
log_test_recall=round(df.loc["1"][1],2)
log_test_f1=round(df.loc["1"][2],2)
print ('log_test_precision ',log_test_precision)
print ('log_test_recall ',log_test_recall)
print ('log_test_f1 ',log_test_f1)


# In[206]:


#3. Random Forest


# In[207]:


param_grid = {
    'max_depth': [10],## 20,30,40
    'max_features': [11],## 7,8,9
    'min_samples_leaf': [10],## 50,100
    'min_samples_split': [50], ## 60,70
    'n_estimators': [100] ## 100,200
}
rfcl = RandomForestClassifier(random_state=1)
grid_search_rf = GridSearchCV(estimator = rfcl, param_grid = param_grid, cv = 5)


# In[208]:


grid_search_rf.fit(X_train, Y_train)


# In[209]:


grid_search_rf.best_params_


# In[210]:


best_grid_rf = grid_search_rf.best_estimator_
best_grid_rf


# In[211]:


# To understand the differences of different random states affecting Out-of-Bag score
random_state=[0,30,64]
for i in random_state:
    rfcl=RandomForestClassifier(random_state=i,oob_score=True)
    rfcl.fit(X_train,Y_train)
    print(rfcl.oob_score_)


# In[212]:


rfcl=RandomForestClassifier(n_estimators=500,random_state=1,oob_score=True,n_jobs=-1)
rfcl=rfcl.fit(X_train,Y_train)
rfcl.oob_score


# In[213]:


rfcl=rfcl.fit(X_test,Y_test)
rfcl.oob_score_


# In[214]:


#3. (a) Model Prediction


# In[215]:


ytrain_predict = best_grid_rf.predict(X_train)
ytest_predict = best_grid_rf.predict(X_test)
ytrain_predict_prob=best_grid_rf.predict_proba(X_train)
pd.DataFrame(ytrain_predict_prob).head()


# In[216]:


ytest_predict_prob=best_grid_rf.predict_proba(X_test)
pd.DataFrame(ytest_predict_prob).head()


# In[217]:


#3. (b) Model Performance


# In[218]:


confusion_matrix(Y_train,ytrain_predict)


# In[219]:


confusion_matrix(Y_test, ytest_predict)


# In[220]:


rf_train_acc=best_grid.score(X_train,Y_train) 
rf_train_acc


# In[221]:


rf_test_acc=best_grid.score(X_test,Y_test)
rf_test_acc


# In[222]:


print(classification_report(Y_train,ytrain_predict))


# In[223]:


print(classification_report(Y_test, ytest_predict))


# In[224]:


#3. (c) ROC-AUC Graph


# In[225]:


rf_train_fpr, rf_train_tpr,_=roc_curve(Y_train,best_grid.predict_proba(X_train)[:,1])
plt.plot(rf_train_fpr,rf_train_tpr,color='green')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
rf_train_auc=roc_auc_score(Y_train,best_grid.predict_proba(X_train)[:,1])
print('Area under Curve is', rf_train_auc)


# In[226]:


rf_test_fpr, rf_test_tpr,_=roc_curve(Y_test,best_grid.predict_proba(X_test)[:,1])
plt.plot(rf_test_fpr,rf_test_tpr,color='green')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
rf_test_auc=roc_auc_score(Y_test,best_grid.predict_proba(X_test)[:,1])
print('Area under Curve is', rf_test_auc)


# In[227]:


#3. (d) Model Performance Metrices


# In[228]:


rf_metrics=classification_report(Y_train, ytrain_predict,output_dict=True)
df=pd.DataFrame(rf_metrics).transpose()
rf_train_precision=round(df.loc["1"][0],2)
rf_train_recall=round(df.loc["1"][1],2)
rf_train_f1=round(df.loc["1"][2],2)
print ('rf_train_precision ',rf_train_precision)
print ('rf_train_recall ',rf_train_recall)
print ('rf_train_f1 ',rf_train_f1)


# In[229]:


rf_metrics=classification_report(Y_test, ytest_predict,output_dict=True)
df=pd.DataFrame(rf_metrics).transpose()
rf_test_precision=round(df.loc["1"][0],2)
rf_test_recall=round(df.loc["1"][1],2)
rf_test_f1=round(df.loc["1"][2],2)
print ('rf_test_precision ',rf_test_precision)
print ('rf_test_recall ',rf_test_recall)
print ('rf_test_f1 ',rf_test_f1)


# In[230]:


#3.(e) Feature Importance


# In[231]:


rf_imp = pd.DataFrame(best_grid_rf.feature_importances_, columns = ["Imp"], 
                      index = X_train.columns).sort_values('Imp',ascending=False)
print(rf_imp)


# In[232]:


#4. Linear Discriminant Analysis


# In[233]:


clf = LinearDiscriminantAnalysis(solver='eigen',shrinkage = 'auto',)
model=clf.fit(X_train,Y_train)
model


# In[234]:


#4. (a) Model Prediction


# In[235]:


# Training Data Class Prediction with a cut-off value of 0.5
pred_class_train = model.predict(X_train)

# Test Data Class Prediction with a cut-off value of 0.5
pred_class_test = model.predict(X_test)
pred_class_test


# In[236]:


# Training Data Probability Prediction
pred_prob_train = model.predict_proba(X_train)

# Test Data Probability Prediction
pred_prob_test = model.predict_proba(X_test)


# In[237]:


#4. (b) Model Performance


# In[238]:


confusion_matrix(Y_train, pred_class_train)


# In[239]:


confusion_matrix(Y_test, pred_class_test)


# In[240]:


lda_train_acc = model.score(X_train,Y_train)
lda_train_acc


# In[241]:


lda_test_acc = model.score(X_test,Y_test)
lda_test_acc


# In[242]:


print(classification_report(Y_train, pred_class_train))


# In[243]:


print(classification_report(Y_test, pred_class_test))


# In[244]:


#4. (c) ROC-AUC Graph


# In[245]:


# AUC and ROC for the training data

# predict probabilities
probs = model.predict_proba(X_train)

# keep probabilities for the positive outcome only
probs = probs[:, 1]

# calculate AUC
from sklearn.metrics import roc_auc_score
lda_train_auc = roc_auc_score(Y_train, probs)
print('AUC: %.3f' % cart_train_auc)

# calculate roc curve
from sklearn.metrics import roc_curve
lda_train_fpr, lda_train_tpr, lda_train_thresholds = roc_curve(Y_train, probs)
plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model
plt.plot(lda_train_fpr, lda_train_tpr, marker='.')

# show the plot
plt.show()


# In[246]:


# AUC and ROC for the test data

# predict probabilities
probs = model.predict_proba(X_test)

# keep probabilities for the positive outcome only
probs = probs[:, 1]

# calculate AUC
from sklearn.metrics import roc_auc_score
lda_test_auc = roc_auc_score(Y_test, probs)
print('AUC: %.3f' % cart_test_auc)

# calculate roc curve
from sklearn.metrics import roc_curve
lda_test_fpr, lda_test_tpr, lda_test_thresholds = roc_curve(Y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model
plt.plot(lda_test_fpr, lda_test_tpr, marker='.')

# show the plot
plt.show()


# In[247]:


#4. (d) Model Performance Metrices


# In[248]:


lda_metrics=classification_report(Y_train, pred_class_train,output_dict=True)
df=pd.DataFrame(lda_metrics).transpose()
lda_train_f1=round(df.loc["1"][2],2)
lda_train_recall=round(df.loc["1"][1],2)
lda_train_precision=round(df.loc["1"][0],2)
print ('lda_train_precision ',lda_train_precision)
print ('lda_train_recall ',lda_train_recall)
print ('lda_train_f1 ',lda_train_f1)


# In[249]:


lda_metrics=classification_report(Y_test, pred_class_test,output_dict=True)
df=pd.DataFrame(lda_metrics).transpose()
lda_test_f1=round(df.loc["0"][2],2)
lda_test_recall=round(df.loc["0"][1],2)
lda_test_precision=round(df.loc["0"][0],2)
print ('lda_test_precision ',lda_test_precision)
print ('lda_test_recall ',lda_test_recall)
print ('lda_test_f1 ',lda_test_f1)


# In[250]:


#5. K Nearest Neighbours


# In[251]:


KNN_model=KNeighborsClassifier(n_neighbors = 15,
                              weights = 'uniform',
                              metric = 'minkowski')
KNN_model.fit(X_train,Y_train)


# In[252]:


#5. (a) Model Prediction


# In[253]:


KNN_train_predict = KNN_model.predict(X_train)
KNN_train_acc = KNN_model.score(X_train, Y_train)
KNN_train_acc


# In[254]:


KNN_test_predict = KNN_model.predict(X_test)
KNN_test_acc = KNN_model.score(X_test, Y_test)
KNN_test_acc


# In[255]:


#5. (b) Model Performance


# In[256]:


confusion_matrix(Y_train, KNN_train_predict)


# In[257]:


confusion_matrix(Y_test, KNN_test_predict)


# In[258]:


print(classification_report(Y_train, KNN_train_predict))


# In[259]:


print(classification_report(Y_test, KNN_test_predict))


# In[260]:


#5. (c) ROC-AUC Graph


# In[261]:


KNN_train_fpr, KNN_train_tpr,_=roc_curve(Y_train,KNN_model.predict_proba(X_train)[:,1])
plt.plot(KNN_train_fpr,KNN_train_tpr,color='black')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
KNN_train_auc=roc_auc_score(Y_train,KNN_model.predict_proba(X_train)[:,1])
print('Area under Curve is', KNN_train_auc)


# In[262]:


KNN_test_fpr, KNN_test_tpr,_=roc_curve(Y_test,KNN_model.predict_proba(X_test)[:,1])
plt.plot(KNN_test_fpr,KNN_test_tpr,color='black')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
KNN_test_auc=roc_auc_score(Y_test,KNN_model.predict_proba(X_test)[:,1])
print('Area under Curve is', KNN_test_auc)


# In[263]:


#5. (d) Model Performance Metrices


# In[264]:


KNN_metrics=classification_report(Y_train, KNN_train_predict,output_dict=True)
df=pd.DataFrame(KNN_metrics).transpose()
KNN_train_f1=round(df.loc["1"][2],2)
KNN_train_recall=round(df.loc["1"][1],2)
KNN_train_precision=round(df.loc["1"][0],2)
print ('KNN_train_precision ',KNN_train_precision)
print ('KNN_train_recall ',KNN_train_recall)
print ('KNN_train_f1 ',KNN_train_f1)


# In[265]:


KNN_metrics=classification_report(Y_test, KNN_test_predict,output_dict=True)
df=pd.DataFrame(KNN_metrics).transpose()
KNN_test_f1=round(df.loc["1"][2],2)
KNN_test_recall=round(df.loc["1"][1],2)
KNN_test_precision=round(df.loc["1"][0],2)
print ('KNN_test_precision ',KNN_test_precision)
print ('KNN_test_recall ',KNN_test_recall)
print ('KNN_test_f1 ',KNN_test_f1)


# In[266]:


# empty list that will hold accuracy scores
ac_scores = []
#
# perform accuracy metrics for values from 1,3,5....19
for k in range(1,20,2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    # evaluate test accuracy
    scores = knn.score(X_test, Y_test)
    ac_scores.append(scores)
#
# changing to misclassification error
MCE = [1 - x for x in ac_scores]
MCE


# In[267]:


# plot misclassification error vs k
plt.plot(range(1,20,2), MCE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()


# In[268]:


#6. Gaussian Naive Bayes


# In[269]:


NB_model = GaussianNB()
NB_model.fit(X_train, Y_train)


# In[270]:


#6. (a) Model Prediction


# In[271]:


NB_train_predict = NB_model.predict(X_train)
NB_train_acc = NB_model.score(X_train, Y_train)   
NB_train_acc 


# In[272]:


NB_test_predict = NB_model.predict(X_test)
NB_test_acc = NB_model.score(X_test, Y_test)
NB_test_acc


# In[273]:


#6. (b) Model Performance


# In[274]:


confusion_matrix(Y_train, NB_train_predict)


# In[275]:


confusion_matrix(Y_test, NB_test_predict)


# In[276]:


print(classification_report(Y_train, NB_train_predict))


# In[277]:


print(classification_report(Y_test, NB_test_predict))


# In[278]:


#6. (c) ROC-AUC Graph


# In[279]:


NB_train_fpr, NB_train_tpr,_=roc_curve(Y_train,NB_model.predict_proba(X_train)[:,1])
plt.plot(NB_train_fpr,NB_train_tpr,color='black')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
NB_train_auc=roc_auc_score(Y_train,NB_model.predict_proba(X_train)[:,1])
print('Area under Curve is', NB_train_auc)


# In[280]:


NB_test_fpr, NB_test_tpr,_=roc_curve(Y_test,NB_model.predict_proba(X_test)[:,1])
plt.plot(NB_test_fpr,NB_test_tpr,color='black')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
NB_test_auc=roc_auc_score(Y_test,NB_model.predict_proba(X_test)[:,1])
print('Area under Curve is', NB_test_auc)


# In[281]:


#6. (d) Model Performance Metrices


# In[282]:


NB_metrics=classification_report(Y_train, NB_train_predict,output_dict=True)
df=pd.DataFrame(NB_metrics).transpose()
NB_train_f1=round(df.loc["1"][2],2)
NB_train_recall=round(df.loc["1"][1],2)
NB_train_precision=round(df.loc["1"][0],2)
print ('NB_train_precision ',NB_train_precision)
print ('NB_train_recall ',NB_train_recall)
print ('NB_train_f1 ',NB_train_f1)


# In[283]:


NB_metrics=classification_report(Y_test, NB_test_predict,output_dict=True)
df=pd.DataFrame(NB_metrics).transpose()
NB_test_f1=round(df.loc["1"][2],2)
NB_test_recall=round(df.loc["1"][1],2)
NB_test_precision=round(df.loc["1"][0],2)
print ('NB_test_precision ',NB_test_precision)
print ('NB_test_recall ',NB_test_recall)
print ('NB_test_f1 ',NB_test_f1)


# In[284]:


#8 Bagging


# In[285]:


#7. Gradient Boosting


# In[286]:


from sklearn.ensemble import GradientBoostingClassifier
Gradient_model = GradientBoostingClassifier(random_state=1)
Gradient_model.fit(X_train,Y_train)


# In[287]:


gbcl_train_acc = Gradient_model.score(X_train,Y_train)
gbcl_train_acc


# In[288]:


gbcl_test_acc =  Gradient_model.score(X_test,Y_test)
gbcl_test_acc


# In[289]:


Y_train_predict7=Gradient_model.predict(X_train)
Gradient_model_score=Gradient_model.score(X_train,Y_train)
print(Gradient_model_score)
print(metrics.confusion_matrix(Y_train,Y_train_predict7))
print(metrics.classification_report(Y_train,Y_train_predict7))


# In[290]:


probs =Gradient_model.predict_proba(X_train)

probs =probs[:,1]

gbcl_train_auc=roc_auc_score(Y_train,probs)
print("the auc curve %.3f " % gbcl_train_auc)

gbcl_train_fpr,gbcl_train_tpr,train_threshold=roc_curve(Y_train,probs)
plt.plot([0,1],[0,1],linestyle='--')

plt.plot(gbcl_train_fpr,gbcl_train_tpr);


# In[291]:



Y_test_predict_7=Gradient_model.predict(X_test)
Graident_model_score=Gradient_model.score(X_test,Y_test)
print(Gradient_model_score)
print(metrics.confusion_matrix(Y_test,Y_test_predict_7))
print(metrics.classification_report(Y_test,Y_test_predict_7))


# In[296]:


#AUC and Roc for test data

probs =Gradient_model.predict_proba(X_test)

probs =probs[:,1]

gbcl_test_auc=roc_auc_score(Y_test,probs)
print("the auc curve %.3f " % gbcl_test_auc)

gbcl_test_fpr,gbcl_test_tpr,test_threshold=roc_curve(Y_test,probs)
plt.plot([0,1],[0,1],linestyle='--')

plt.plot(gbcl_test_fpr, gbcl_test_tpr);


# In[297]:


gbcl_metrics=classification_report(Y_train, Y_train_predict7,output_dict=True)
df=pd.DataFrame(gbcl_metrics).transpose()
gbcl_train_f1=round(df.loc["1"][2],2)
gbcl_train_recall=round(df.loc["1"][1],2)
gbcl_train_precision=round(df.loc["1"][0],2)
gbcl_train_acc=round(df.iloc[2][3],2)
print('Gradient_train_accuracy',gbcl_train_acc)
print ('gbcl_train_precision ',gbcl_train_precision)
print ('gbcl_train_recall ',gbcl_train_recall)
print ('gbcl_train_f1 ',gbcl_train_f1)


# In[298]:


#print(metrics.classification_report(Y_test,Y_test_predict_7))
gbcl_metrics=classification_report(Y_test,Y_test_predict_7,output_dict=True)
df=pd.DataFrame(gbcl_metrics).transpose()
gbcl_test_f1=round(df.loc["1"][2],2)
gbcl_test_recall=round(df.loc["1"][1],2)
gbcl_test_precision=round(df.loc["1"][0],2)
print ('gbcl_test_precision ',gbcl_test_precision)
print ('gbcl_test_recall ',gbcl_test_recall)
print ('gbcl_test_f1 ',gbcl_test_f1)


# In[299]:


##MLP Classifier (Artificial Neural Network)


# In[300]:


param_grid = {
    'hidden_layer_sizes': [(100,100,100)],
    'activation': ['logistic', 'relu'],
    'solver': ['sgd', 'adam'],
    'tol': [0.1,0.01],
    'max_iter' : [10000]
}

mlp = MLPClassifier()

grid_search = GridSearchCV(estimator = mlp, param_grid = param_grid, cv = 3)


# In[301]:


#ytrain_predict = mlp.predict(X_train)
#ytest_predict = mlp.predict(X_test)

grid_search.fit(X_train, Y_train)
grid_search.best_params_


# In[302]:


best_grid = grid_search.best_estimator_
ann_train_predict_prob=best_grid.predict_proba(X_train)
pd.DataFrame(ann_train_predict_prob).head()


# In[303]:


ann_test_predict_prob=best_grid.predict_proba(X_test)
pd.DataFrame(ann_test_predict_prob).head()


# In[304]:



ytrain_predict = best_grid.predict(X_train)
ytest_predict = best_grid.predict(X_test)


# In[305]:


#Model performance


# In[306]:


ann_train_acc = best_grid.score(X_train,Y_train)
ann_train_acc


# In[307]:


ann_test_acc = best_grid.score(X_test,Y_test)
ann_test_acc


# In[308]:



print(classification_report(Y_train,ytrain_predict))


# In[309]:


print(classification_report(Y_test,ytest_predict))


# In[310]:


confusion_matrix(Y_train, ytrain_predict) 


# In[311]:


confusion_matrix(Y_test, ytest_predict)


# In[312]:


#ROC-AUC Graph train


# In[313]:


# AUC and ROC for the training data

# predict probabilities
probs =best_grid.predict_proba(X_train)

# keep probabilities for the positive outcome only
probs = probs[:, 1]

# calculate AUC
from sklearn.metrics import roc_auc_score
ann_train_auc = roc_auc_score(Y_train, probs)
print('AUC: %.3f' % ann_train_auc)

# calculate roc curve
from sklearn.metrics import roc_curve
ann_train_fpr, ann_train_tpr, ann_train_thresholds = roc_curve(Y_train, probs)
plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model
plt.plot(ann_train_fpr, ann_train_tpr, marker='.')

# show the plot
plt.show()


# In[314]:


# AUC and ROC for the testing data

# predict probabilities
probs =best_grid.predict_proba(X_test)

# keep probabilities for the positive outcome only
probs = probs[:, 1]

# calculate AUC
from sklearn.metrics import roc_auc_score
ann_test_auc = roc_auc_score(Y_test, probs)
print('AUC: %.3f' % ann_test_auc)

# calculate roc curve
from sklearn.metrics import roc_curve
ann_test_fpr, ann_test_tpr, ann_test_thresholds = roc_curve(Y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model
plt.plot(ann_test_fpr, ann_test_tpr, marker='.')

# show the plot
plt.show()


# In[315]:


# Model Performance Metrices


# In[316]:


ann_metrics=classification_report(Y_train, ytrain_predict,output_dict=True)
df=pd.DataFrame(ann_metrics).transpose()
ann_train_f1=round(df.loc["1"][2],2)
ann_train_recall=round(df.loc["1"][1],2)
ann_train_precision=round(df.loc["1"][0],2)
print ('ann_train_precision ',ann_train_precision)
print ('ann_train_recall ',ann_train_recall)
print ('ann_train_f1 ',ann_train_f1)


# In[317]:


ann_metrics=classification_report(Y_test, ytest_predict,output_dict=True)
df=pd.DataFrame(ann_metrics).transpose()
ann_test_f1=round(df.loc["1"][2],2)
ann_test_recall=round(df.loc["1"][1],2)
ann_test_precision=round(df.loc["1"][0],2)
print ('ann_test_precision ',ann_test_precision)
print ('ann_test_recall ',ann_test_recall)
print ('ann_test_f1 ',ann_test_f1)


# In[318]:


#Model Performance Comparison


# In[319]:


#1. Performance Matrics - on Train Data


# In[320]:


index=['Accuracy', 'AUC', 'Recall','Precision','F1 Score']
data = pd.DataFrame({
       'CART train':[cart_train_acc,cart_train_auc,cart_train_recall,cart_train_precision,cart_train_f1],
       'RF Train':[rf_train_acc,rf_train_auc,rf_train_recall,rf_train_precision,rf_train_f1],
       'Log Train':[log_train_acc,log_train_auc,log_train_recall,log_train_precision,log_train_f1],          
       'LDA Train':[lda_train_acc,lda_train_auc,lda_train_recall,lda_train_precision,lda_train_f1],
       'KNN Train':[KNN_train_acc,KNN_train_auc,KNN_train_recall,KNN_train_precision,KNN_train_f1], 
       'NB Train':[NB_train_acc,NB_train_auc,NB_train_recall,NB_train_precision,NB_train_f1],
       'ANN train':[ann_train_acc,ann_train_auc,ann_train_recall,ann_train_precision,ann_train_f1],
       'Gr.Boost Train':[gbcl_train_acc,gbcl_train_auc,gbcl_train_recall,gbcl_train_precision,gbcl_train_f1]},index=index)
round(data,2)


# In[321]:


#2. Performance Matrics - on Test Data


# In[322]:


index=['Accuracy', 'AUC', 'Recall','Precision','F1 Score']
data = pd.DataFrame({
       'CART test':[cart_test_acc,cart_test_auc,cart_test_recall,cart_test_precision,cart_test_f1],
       'RF test':[rf_test_acc,rf_test_auc,rf_test_recall,rf_test_precision,rf_test_f1],  
       'Log Test':[log_test_acc,log_test_auc,log_test_recall,log_test_precision,log_test_f1],
       'LDA test':[lda_test_acc,lda_test_auc,lda_test_recall,lda_test_precision,lda_test_f1],
       'KNN test':[KNN_test_acc,KNN_test_auc,KNN_test_recall,KNN_test_precision,KNN_test_f1], 
       'NB test':[NB_test_acc,NB_test_auc,NB_test_recall,NB_test_precision,NB_test_f1],
       'ANN test':[ann_test_acc,ann_test_auc,ann_test_recall,ann_test_precision,ann_test_f1],
       'Gr.Boost test':[gbcl_test_acc,gbcl_test_auc,gbcl_test_recall,gbcl_test_precision,gbcl_test_f1]},index=index)
round(data,2)


# In[323]:


#3. ROC-AUC - on Train Data


# In[324]:


plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(cart_train_fpr, cart_train_tpr,color='red',label="CART")
plt.plot(ann_train_fpr, ann_train_tpr,color='orange',label="ANN")
plt.plot(rf_train_fpr,rf_train_tpr,color='green',label="RF")
plt.plot(log_train_fpr,log_train_tpr,color='Pink',label="Logistic")
plt.plot(KNN_train_fpr,KNN_train_tpr,color='magenta',label="KNN")
plt.plot(NB_train_fpr,NB_train_tpr,color='yellow',label="NB")
plt.plot(lda_train_fpr,lda_train_tpr,color='blue',label="LDA")
plt.plot(gbcl_train_fpr,gbcl_train_tpr,color='violet',label="Gr. Boost")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower right')


# In[325]:


#4. ROC-AUC - on Test Data


# In[326]:


plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(cart_test_fpr, cart_test_tpr,color='red',label="CART")
plt.plot(rf_test_fpr,rf_test_tpr,color='green',label="RF")
plt.plot(log_test_fpr,log_test_tpr,color='pink',label="Logistic")
plt.plot(lda_test_fpr,lda_test_tpr,color='magenta',label="LDA")
plt.plot(KNN_test_fpr,KNN_test_tpr,color='yellow',label="KNN")
plt.plot(NB_test_fpr,NB_test_tpr,color='blue',label="NB")
plt.plot(gbcl_test_fpr,gbcl_test_tpr,color='violet',label="Gr. Boost")
#plt.plot(xgb_test_fpr,xgb_test_tpr,color='black',label="XG Boost")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower right')


# In[ ]:


# Ensembling -Bagging


# In[46]:



from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
cart = DecisionTreeClassifier()
Bagging_model=BaggingClassifier(base_estimator=cart,n_estimators=100,random_state=1)
Bagging_model.fit(X_train, Y_train)


# In[57]:



# Training Data Probability Prediction
pred_prob_train = Bagging_model.predict_proba(X_train)

# Test Data Probability Prediction
pred_prob_test = Bagging_model.predict_proba(X_test)


# In[47]:



## Performance Matrix on train data set
Y_train_predict = Bagging_model.predict(X_train)
model_score =Bagging_model.score(X_train, Y_train)
print(model_score)
print(metrics.confusion_matrix(Y_train, Y_train_predict))
print(metrics.classification_report(Y_train, Y_train_predict))


# In[48]:



## Performance Matrix on test data set
Y_test_predict = Bagging_model.predict(X_test)
model_score = Bagging_model.score(X_test, Y_test)
print(model_score)
print(metrics.confusion_matrix(Y_test, Y_test_predict))
print(metrics.classification_report(Y_test, Y_test_predict))


# In[65]:


enasable_bagg_train_acc=Bagging_model.score(X_train,Y_train) 
enasable_bagg_train_acc


# In[66]:


ensamble_bagg_test_acc=Bagging_model.score(X_test,Y_test) 
ensamble_bagg_test_acc


# In[59]:


# Ensembling -Random Forest


# In[62]:


from sklearn.ensemble import RandomForestClassifier

RF_model=RandomForestClassifier(n_estimators=100,random_state=1)
RF_model.fit(X_train, Y_train)


# In[63]:



## Performance Matrix on train data set
Y_train_predict = RF_model.predict(X_train)
model_score =RF_model.score(X_train, Y_train)
print(model_score)
print(metrics.confusion_matrix(Y_train, Y_train_predict))
print(metrics.classification_report(Y_train, Y_train_predict))


# In[64]:



## Performance Matrix on test data set
Y_test_predict = RF_model.predict(X_test)
model_score = RF_model.score(X_test, Y_test)
print(model_score)
print(metrics.confusion_matrix(Y_test, Y_test_predict))
print(metrics.classification_report(Y_test, Y_test_predict))


# In[68]:


enasable_rf_train_acc=RF_model.score(X_train,Y_train) 
enasable_rf_train_acc


# In[69]:


enasable_rf_test_acc=RF_model.score(X_test,Y_test) 
enasable_rf_test_acc


# In[ ]:




