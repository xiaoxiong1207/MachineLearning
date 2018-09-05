
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from scipy import stats
import operator
import timeit
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import datasets
from matplotlib.pyplot import figure
from scipy.stats import chi2_contingency

from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch


# In[3]:


start = timeit.default_timer()

DnB_F=pd.read_csv('D:/Xiaoxi/ModelC/Dataset/Model_Transactions_201807_PPSS_deduped_modelc.csv')
DnB_F.head()
DnB_F=pd.DataFrame(DnB_F)

stop = timeit.default_timer()
print ('runing time:',round((stop - start)/60,2),'mins')


# In[4]:


pd.set_option("display.max_columns", 150)
pd.set_option("display.max_rows", 50)


# In[5]:


DnB_F.columns


# In[10]:


len(DnB_F)


# In[11]:


DnB_F


# In[12]:


len(DnB_F.columns)  # 68 columns


# In[13]:


## Frequency table

print(DnB_F.QUALITY_RESPONSECOUNT.value_counts())
print(DnB_F.FUND_CNT.value_counts())


# In[14]:


####### Copy the dataset  ##########################################
DnB_F1=DnB_F.copy()

print (len(DnB_F1))

# ONLY SELECT ADDRESS_TYPE='PRACTICE','Practice','Mailing','TU HOME3'

DnB_F1=DnB_F1[DnB_F1['ADDRESS_TYPE'].isin(['PRACTICE','Practice','Mailing','TU HOME3','FA BUSINESS','CPA BUSINESS'])]

DnB_F1=DnB_F1[DnB_F1['SPECIALTY'].isin(['PRIMARY','SECONDARY','BHG PRO'])]


DnB_F1['ADDRESS_TYPE'].replace('PRACTICE','Practice',inplace=True)

print('Filter ADDRESS_TYPE AND SPECIALTY:',len(DnB_F1))


# In[15]:


DnB_F1.FICO.count()


# In[16]:


DnB_F1.ADDRESS_TYPE.value_counts()


# ## Recoding Medical_title and CHIEF_EXECUTIVE_OFFICER_-_TITLE_r

# In[309]:


####  MedicalTitle Recode using MedicalTitle Group ####

# DnB_F1.MedicalTitle.value_counts()
# pd.crosstab(DnB_F1['MedicalTitle'], columns="count")


# In[17]:


def Medical_title(series):
  
    if series in (['DO','DPM','OD']):
        return 'MD'
    elif series in (['DMD','DN','DENT']):
        return 'DDS'
    elif series in(['DVM','VMD','VET']):
        return 'VET'
    elif series in (['APRN','LPN','TRN']):
        return 'RN'
    elif series in (['PharmD','PHARMD']):
        return 'RPH'
    elif series in (['ANP','ARNP','CNP','CPNP','CRNP','FNP','GNP','PNP','WHNP']):
        return 'NP'
    elif series in (['PAC']):
        return 'PA'  
    elif series in (['ND']):
        return 'DC'
    elif series in (['PSO']):
        return 'PSYD'
    else:
        return series


# In[18]:


## Create MedicalTitle into MedicalTitleGroup

DnB_F1['MedicalTitleGroup'] = DnB_F1['MedicalTitle'].apply(Medical_title)

DnB_F1['MedicalTitleGroup'].value_counts()


# In[19]:


def Medical_title_group_reduce(series):
  
    if series in (['MD']):
        return 'MD'
    elif series in(['NP']):
        return 'NP'    
    elif series in (['RPH']):
        return 'RPH'
    elif series in (['DDS']):
        return 'DDS'
    elif series in (['FA']):
        return 'FA'
    elif series in (['PA']):
        return 'PA'  
    elif series in (['PT']):
        return 'PT'    
    elif series in (['SLP']):
        return 'SLP'
    elif series in (['CPA']):
        return 'CPA'  
    elif series in (['OT']):
        return 'OT'
    elif series in (['CRNA']):
        return 'CRNA'
    else:
        return 'OTHER'


# In[20]:


DnB_F1['MedicalTitleGroup_r'] = DnB_F1['MedicalTitleGroup'].apply(Medical_title_group_reduce)
DnB_F1['MedicalTitleGroup_r'].value_counts()


# In[21]:


def Region(series):
  
    if series in (['CA','NV','UT','CO','WY','ID','MT','WA','OR','AK','HI']):
        return 'west'
    elif series in(['AZ','NM','TX','OK']):
        return 'southwest'    
    elif series in (['ND','SD','NE','KS','MN','IA','MO','WI','IL','MI','IN','OH']):
        return 'midwest'
    elif series in (['MD','PA','NY','NJ','CT','RI','MA','NH','VT','ME']):
        return 'northeast'
    elif series in (['AR','LA','MS','AL','TN','KY','GA','FL','SC','NC','VA','DC','WV','DE']):
        return 'southeast'


# In[22]:


DnB_F1['Region'] = DnB_F1['State'].apply(Region)
DnB_F1['Region'].value_counts()


# In[23]:


def Recode3(series):
  
    if series <=2:
        return series
    elif series >2:
        return 


# In[24]:


def Recode4(series):
  
    if series <=3:
        return series
    elif series >3:
        return 4


# In[25]:


def Recode5(series):
  
    if series <=4:
        return series
    elif series >4:
        return 5


# In[26]:


def Recode7(series):
  
    if series <=6:
        return series
    elif series >6:
        return 7


# In[27]:


DnB_F1['Num_Business_r']=DnB_F1['Num_Business'].apply(Recode3)
DnB_F1['Num_Home_r']=DnB_F1['Num_Home'].apply(Recode3)
DnB_F1['SENT_3MO_r']=DnB_F1['SENT_3MO'].apply(Recode3)
DnB_F1['SENT_6MO_r']=DnB_F1['SENT_6MO'].apply(Recode4)
DnB_F1['SENT_12MO_r']=DnB_F1['SENT_12MO'].apply(Recode5)
DnB_F1['SENT_18MO_r']=DnB_F1['SENT_18MO'].apply(Recode5)
DnB_F1['SENT_24MO_r']=DnB_F1['SENT_24MO'].apply(Recode5)
DnB_F1['Total_Sent_Practitioner_r']=DnB_F1['Total_Sent_Practitioner'].apply(Recode7)
DnB_F1['Total_Sent_Site_r']=DnB_F1['Total_Sent_Site'].apply(Recode7)


# In[28]:


DnB_F1['FICO_r']=pd.qcut(DnB_F1['FICO'], q=10,duplicates='drop',labels=[1,2,3,4,5,6,7,8,9,10])


# In[29]:


#col_y = ['QUALITY_RESPONSECOUNT']  
col_y = ['FUND_CNT'] 


# In[30]:


# Create category and continous variable for MODEL

col_categ=[
'MedicalTitle',
'MedicalTitleGroup',
'MedicalTitleGroup_r',
'SPECIALTY',
'TU_GROUP',
#'Previous_TU_GROUP',
'Decile',
'ADDRESS_TYPE',
'Site_type',
#'SiteFirstUse',
'Region',
#'Date',
#'LastTransactionDate',
#'LastTransaction_Days',
#'MailOrder',
#'Total_SENT_Month',
'NEWTU',
#'Shipping',
#'Product',
#'Sent',
#'RFL',
#'Bad_Data',
#'Non_HealthCare',
'MODEL',
'New_Practitioner',
'New_Site',
'FICO_r',
]


# In[31]:


col_contin=[
'Num_Home_r',
'Num_Business_r',
'Total_Sent_Practitioner_r',
'Total_Sent_Site_r',
'SENT_3MO_r',
'SENT_6MO_r',
'SENT_12MO_r',
'SENT_18MO_r',
'SENT_24MO_r',
#'FICO_r',
'FICO'
 ]


# In[32]:


print('col_categ', col_categ)
print('******************************************************************************************************')
print('col_contin', col_contin)

#DnB_F1[col_categ].describe(include=[np.object])
#DnB_F1[col_contin].describe()


# In[33]:


# print(DnB_F[col_contin].isnull().sum())  # Count missing value
print('******************************************************************************************************')
print(DnB_F1[col_contin].isnull().sum())  # Count missing value
print('******************************************************************************************************')
print(DnB_F1[col_categ].isnull().sum())  # Count missing value


# In[34]:


pd.crosstab(pd.cut(DnB_F1['NEWMARGIN'],bins=5),DnB_F1['FUND_CNT'],dropna=False)


# In[35]:


## Make copy of dataset to aviod issue

DnB_F1_contin =DnB_F1[col_contin].copy()
DnB_F1_categ = DnB_F1[col_categ].copy()

## Concatenating dataframes together by columns

#DnB_F_model=pd.concat([DnB_F1_contin,DnB_F1_categ,DnB_F1['FUND_CNT'],DnB_F1['QUALITY_RESPONSECOUNT']],axis=1)    
DnB_F_model=pd.concat([DnB_F1_contin,DnB_F1_categ,DnB_F1['QUALITY_RESPONSECOUNT'],DnB_F1['FUND_CNT'],DnB_F1['NEWMARGIN'],DnB_F1['PID'],DnB_F1['POWNER']],axis=1)


# In[36]:


DnB_F1['tr_year'] = pd.DatetimeIndex(DnB_F1['Date']).year
DnB_F1['tr_month'] = pd.DatetimeIndex(DnB_F1['Date']).month
DnB_F1['tr_day'] = pd.DatetimeIndex(DnB_F1['Date']).day


# In[37]:


DnB_F1['tr_year'].value_counts()


# In[38]:


pd.crosstab(DnB_F1['tr_year'],DnB_F1['tr_month'])


# In[70]:


# seperate data into 2016.6 and 2018.1-3 to predict 2018.4-7

df_modelc_P1 = DnB_F_model[(DnB_F1.tr_year == 2016)]
df_modelc_P2 = DnB_F_model[(DnB_F1.tr_year == 2017)]
df_modelc_P3 = DnB_F_model[(DnB_F1.tr_year == 2018) & (DnB_F1.tr_month<=5)]

df_modelc_P=pd.concat([df_modelc_P1,df_modelc_P2,df_modelc_P3]) # Concatenating two dataframe together by rows

df_modelc_P_201804 = DnB_F_model[(DnB_F1.tr_year == 2018) & (DnB_F1.tr_month==4)]
df_modelc_P_201805 = DnB_F_model[(DnB_F1.tr_year == 2018) & (DnB_F1.tr_month==5)]
df_modelc_P_201806 = DnB_F_model[(DnB_F1.tr_year == 2018) & (DnB_F1.tr_month==6)]
df_modelc_P_201807 = DnB_F_model[(DnB_F1.tr_year == 2018) & (DnB_F1.tr_month==7)]


# In[71]:


# save python dataframe into csv

df_modelc_P.to_csv('D:/Xiaoxi/ModelC/Dataset/csv/GBM/df_modelc_P_frequency_p.csv')

df_modelc_P_201804.to_csv('D:/Xiaoxi/ModelC/Dataset/csv/GBM/df_modelc_P_201804_frequency_p.csv')
df_modelc_P_201805.to_csv('D:/Xiaoxi/ModelC/Dataset/csv/GBM/df_modelc_P_201805_frequency_p.csv')
df_modelc_P_201806.to_csv('D:/Xiaoxi/ModelC/Dataset/csv/GBM/df_modelc_P_201806_frequency_p.csv')
df_modelc_P_201807.to_csv('D:/Xiaoxi/ModelC/Dataset/csv/GBM/df_modelc_P_201807_frequency_p.csv')


# ## H2O

# In[42]:


import h2o
# Initialize H2O using h2o.init().
h2o.init()


# In[72]:


# read dataset from csv

h2o_df_P=h2o.import_file('D:/Xiaoxi/ModelC/Dataset/csv/GBM/df_modelc_P_frequency_p.csv')

h2o_df_P_201804=h2o.import_file('D:/Xiaoxi/ModelC/Dataset/csv/GBM/df_modelc_P_201804_frequency_p.csv')
h2o_df_P_201805=h2o.import_file('D:/Xiaoxi/ModelC/Dataset/csv/GBM/df_modelc_P_201805_frequency_p.csv')
h2o_df_P_201806=h2o.import_file('D:/Xiaoxi/ModelC/Dataset/csv/GBM/df_modelc_P_201806_frequency_p.csv')
h2o_df_P_201807=h2o.import_file('D:/Xiaoxi/ModelC/Dataset/csv/GBM/df_modelc_P_201807_frequency_p.csv')


# In[73]:


# DnB_F_model 
DnB_F_model.to_csv('D:/Xiaoxi/ModelC/Dataset/csv/GBM/df_modelc_frequency_p.csv')

# Create H2O dataset

h2o_DnB_F_model=h2o.import_file('D:/Xiaoxi/ModelC/Dataset/csv/GBM/df_modelc_frequency_p.csv')

h2o_DnB_F_model_df=h2o_DnB_F_model.as_data_frame()  


# In[74]:


# Change data type

for i in col_categ: 
    h2o_df_P[i] = h2o_df_P[i].asfactor()
    h2o_df_P_201804[i] = h2o_df_P_201804[i].asfactor()
    h2o_df_P_201805[i] = h2o_df_P_201805[i].asfactor()
    h2o_df_P_201806[i] = h2o_df_P_201806[i].asfactor()
    h2o_df_P_201807[i] = h2o_df_P_201807[i].asfactor()
    
for i in col_contin: 
    h2o_df_P[i] = h2o_df_P[i].asnumeric()
    h2o_df_P_201804[i] = h2o_df_P_201804[i].asnumeric()
    h2o_df_P_201805[i] = h2o_df_P_201805[i].asnumeric()
    h2o_df_P_201806[i] = h2o_df_P_201806[i].asnumeric()
    h2o_df_P_201807[i] = h2o_df_P_201807[i].asnumeric()
    
### transfer h2o dataset into pd.dataframe new dataset

h2o_df_P_df=h2o_df_P.as_data_frame()   
 
h2o_df_P_201804_df=h2o_df_P_201804.as_data_frame()  
h2o_df_P_201805_df=h2o_df_P_201805.as_data_frame() 
h2o_df_P_201806_df=h2o_df_P_201806.as_data_frame()    
h2o_df_P_201807_df=h2o_df_P_201807.as_data_frame()     


# In[46]:


col_categ1=[
#'MedicalTitleGroup',
'MedicalTitleGroup_r',
'SPECIALTY',
'TU_GROUP',
#'Previous_TU_GROUP',
'Decile',
'ADDRESS_TYPE',
'Site_type',
'Region',
#'LastTransaction_Days',
#'MailOrder',
#'Total_SENT_Month',
#'NEWTU',
#'Shipping',
#'Product',
#'Sent',
#'RFL',
# 'Bad_Data',
#'Non_HealthCare',
'MODEL',
'New_Practitioner',
'New_Site',
'FICO_r'
]


# In[47]:


col_contin1=[
'Num_Home_r',
'Num_Business_r',
'Total_Sent_Practitioner_r',
'Total_Sent_Site_r',
'SENT_3MO_r',
'SENT_6MO_r',
'SENT_12MO_r',
'SENT_18MO_r',
'SENT_24MO_r',
#'FICO_r'
#'FICO'
 ]


# In[48]:


#col_select1 =  col_contin1  + col_categ1+ col_y  
col_select1 =  col_contin1  + col_categ1+ col_y  


# In[49]:


#y_column = 'QUALITY_RESPONSECOUNT'
y_column = 'FUND_CNT'


X_columns_a =  col_select1.copy()   
#X_columns_a.remove('QUALITY_RESPONSECOUNT')
X_columns_a.remove('FUND_CNT')

X_columns = X_columns_a

train = h2o_df_P
#train[ 'QUALITY_RESPONSECOUNT'] = train[ 'QUALITY_RESPONSECOUNT'].asfactor()
train[ 'FUND_CNT'] = train[ 'FUND_CNT'].asfactor()
print('train shape', train.shape) 

#y = 'QUALITY_RESPONSECOUNT'
y = 'FUND_CNT'
x = X_columns
x


# ## GLM Logistic Regression with Lasso regularization 

# In[52]:


glm_model = H2OGeneralizedLinearEstimator(family= "binomial",
                                          alpha=1,
                                          #lambda_=0.000086,
                                          lambda_search = True, 
                                          #balance_classes=True,
                                          #missing_values_handling="mean_imputation",
                                          #compute_p_values = True,#(this is avalible when regularization is disabled lambda=0) 
                                          remove_collinear_columns=True)

                                          # lambda_search = True, alpha=1 (Lasso penalties) 

########v Validation and Train data  #############################

traindata, valid = train.split_frame(ratios = [.7],seed = 12345)

glm_model.train(x , y, training_frame = traindata, validation_frame = valid)

glm_model


# In[53]:


glm_coefficient=glm_model._model_json['output']['coefficients_table'].as_data_frame()   # some variables are penalties to 0
glm_coefficient

glm_coefficient[(glm_coefficient.standardized_coefficients!=0)]


# In[54]:


glm_coefficient_nonezero=glm_coefficient[(glm_coefficient.standardized_coefficients!=0)][1:].names   
print('Not unique nonezero variable:',glm_coefficient_nonezero.count())
glm_coefficient_nonezero


# In[55]:


unique_list=set(glm_coefficient_nonezero.str.split('.',expand=True)[0])
print('Unique Variable count:',len(unique_list))
unique_list


# ## Predict

# ### In time validation

# In[56]:


perf_glm=glm_model.model_performance(train=True)  # Train ROC AUC 0.7147
perf_glm_valid =glm_model.model_performance(valid)  # Valid ROC AUC 0.6487

perf_glm.plot()
perf_glm_valid.plot()


# ## Out of time validation

# In[57]:


## Self definition to give AUC and KS

def AUC_KS (True_label,Pred_Prob):
    auc=metrics.roc_auc_score(True_label,Pred_Prob)
    print ('\033[1m','AUC is:', round(auc,6))
    fpr, tpr, thresholds = metrics.roc_curve(True_label, Pred_Prob, pos_label=1)
    print('\033[1m','KS is:',round(max(tpr-fpr),5)) 
    plt.plot(fpr,tpr)
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    return

## Self definition to give MAPE for Response Rate

def MAPE(data):
    data_sort=data.sort_values('p1', ascending=False)
    data_sort['bucket']= pd.qcut(data_sort.p1,10, duplicates='drop')  
    grouped = data_sort.groupby('bucket', as_index = False)
    agg=pd.DataFrame(grouped.mean().p1)
    agg=agg.sort_values('p1', ascending = False)
    agg['positive']=grouped.sum().FUND_CNT
    agg['total']=grouped.count().FUND_CNT
    agg['one_rate']=agg['positive']/agg['total']
    agg.columns=['y_pred','n_positives','total','one_rate']
    agg['abs_error']=(abs(agg['one_rate']-agg['y_pred'])/agg['one_rate'])/10
    MAPE=agg.abs_error.sum()
    print ('\033[1m','MAPE is',round(MAPE,6))
    return agg


def Margin(data):
    data_sort=data.sort_values('p1', ascending=False)
    data_sort['bucket']= pd.qcut(data_sort.p1,10, duplicates='drop')  
    grouped = data_sort.groupby('bucket', as_index = False)
    agg=pd.DataFrame(grouped.mean().p1)
    agg=agg.sort_values('p1', ascending = False)
    agg['NEWMARGIN']=round(grouped.sum().NEWMARGIN,1)
    agg['cumMargin']=round(agg['NEWMARGIN'].cumsum(),1)
    agg['Percent']=round((agg['NEWMARGIN']/agg['NEWMARGIN'].sum()).cumsum(),3)
  
    return agg


# #### 201804

# In[75]:


pred_201804=glm_model.predict(h2o_df_P_201804).as_data_frame()

AUC_KS(h2o_df_P_201804_df['FUND_CNT'],pred_201804[['p1']])

data_201804=pd.concat([h2o_df_P_201804_df['FUND_CNT'],pred_201804[['p1']]],axis=1) 

MAPE(data_201804)


# In[76]:


Margin(pd.concat([h2o_df_P_201804_df['FUND_CNT'],pred_201804[['p1']],h2o_df_P_201804_df['NEWMARGIN']],axis=1) )


# #### 201805

# In[77]:


pred_201805=glm_model.predict(h2o_df_P_201805).as_data_frame()

AUC_KS(h2o_df_P_201805_df['FUND_CNT'],pred_201805[['p1']])

data_201805=pd.concat([h2o_df_P_201805_df['FUND_CNT'],pred_201805[['p1']]],axis=1) 

MAPE(data_201805)


# In[78]:


Margin(pd.concat([h2o_df_P_201805_df['FUND_CNT'],pred_201805[['p1']],h2o_df_P_201805_df['NEWMARGIN']],axis=1) )


# #### 201806

# In[62]:


pred_201806=glm_model.predict(h2o_df_P_201806).as_data_frame()

AUC_KS(h2o_df_P_201806_df['FUND_CNT'],pred_201806[['p1']])

data_201806=pd.concat([h2o_df_P_201806_df['FUND_CNT'],pred_201806[['p1']]],axis=1) 

MAPE(data_201806)


# In[63]:


Margin(pd.concat([h2o_df_P_201806_df['FUND_CNT'],pred_201806[['p1']],h2o_df_P_201806_df['NEWMARGIN']],axis=1) )


# #### 201807

# In[64]:


pred_201807=glm_model.predict(h2o_df_P_201807).as_data_frame()

AUC_KS(h2o_df_P_201807_df['FUND_CNT'],pred_201807[['p1']])

data_201807=pd.concat([h2o_df_P_201807_df['FUND_CNT'],pred_201807[['p1']]],axis=1) 

MAPE(data_201807)


# In[65]:


Margin(pd.concat([h2o_df_P_201807_df['FUND_CNT'],pred_201807[['p1']],h2o_df_P_201807_df['NEWMARGIN']],axis=1) )


# In[66]:


pred_all=glm_model.predict(h2o_DnB_F_model).as_data_frame()

AUC_KS(h2o_DnB_F_model_df['FUND_CNT'],pred_all[['p1']])

data_all=pd.concat([h2o_DnB_F_model_df['FUND_CNT'],pred_all[['p1']]],axis=1) 

MAPE(data_all)


# In[67]:


Margin(pd.concat([h2o_DnB_F_model_df['FUND_CNT'],pred_all[['p1']],h2o_DnB_F_model_df['NEWMARGIN']],axis=1) )


# In[68]:


pred_p=glm_model.predict(h2o_df_P).as_data_frame()

AUC_KS(h2o_df_P_df['FUND_CNT'],pred_p[['p1']])

data_all=pd.concat([h2o_df_P_df['FUND_CNT'],pred_p[['p1']]],axis=1) 

MAPE(data_all)


# In[69]:


Margin(pd.concat([h2o_df_P_df['FUND_CNT'],pred_p[['p1']],h2o_df_P_df['NEWMARGIN']],axis=1) )


# # GBM Grid Search

# In[118]:


##### Define parameters for  Gridsearch

hyper_parameters = {'ntrees': [40,60,100,150], 
                    'nbins_cats':[4,8,16,32],
                    'nbins':[8,16,32,64],
                    'nbins_top_level': [64,128],
                    'max_depth':[2,3]
                    }

search_criteria = {'strategy': 'RandomDiscrete', 'max_models': 216, 'seed': 1234}

traindata, valid = train.split_frame(ratios = [.7],seed = 12345)

hyper_parameters


# In[119]:


# Train and validate a cartesian grid of GBMs
start = timeit.default_timer()

gbm_grid1 = H2OGridSearch(model=H2OGradientBoostingEstimator(learn_rate =0.1,col_sample_rate=0.6,sample_rate=0.6),
                          search_criteria=search_criteria,
                          hyper_params=hyper_parameters)


gbm_grid1.train(x, y, training_frame=traindata,validation_frame = valid)


stop = timeit.default_timer()
print ('runing time:',round((stop - start)/60,2),'mins')


# In[120]:


# Get the grid results, sorted by validation AUC
gbm_gridperf1 = gbm_grid1.get_grid(sort_by='auc', decreasing=True)
gbm_gridperf1


# In[148]:


# Get the grid results, sorted by validation AUC
#gbm_gridperf1 = gbm_grid1.get_grid(sort_by='auc', decreasing=True)
#gbm_gridperf1


# ## Best GBM

# In[428]:


traindata, valid = train.split_frame(ratios = [.7],seed = 12345)

######################

learn_rate =0.1
ntrees=60
max_depth = 3
col_sample_rate = 0.6
sample_rate=0.6

nbins_top_level =128
nbins = 32
nbins_cats = 8

#####  GBM  ##############################

best_gbm1 = H2OGradientBoostingEstimator(
   learn_rate=learn_rate,
   ntrees=ntrees,
   max_depth=max_depth,
   sample_rate=sample_rate,
   col_sample_rate=col_sample_rate,
   nbins_top_level =nbins_top_level ,
   nbins=nbins,
   nfolds=3,
   nbins_cats = nbins_cats,
   #calibrate_model = True,
   #calibration_frame = traindata,
   model_id="Model C",
   categorical_encoding="one_hot_explicit", # this one can improve performance
   distribution='bernoulli',
   #balance_classes=True,
   seed=12345
)


# In[429]:


x


# In[430]:


start = timeit.default_timer()

best_gbm1.train(x, y, training_frame=traindata,validation_frame = valid)

stop = timeit.default_timer()
print ('runing time:',round((stop - start)/60,2),'mins')


# In[431]:


best_gbm1


# ### Important variables

# In[432]:


gbm_coefficient=best_gbm1._model_json['output']['variable_importances'].as_data_frame()
gbm_coefficient[(gbm_coefficient.iloc[:,2]!=0)]


# In[433]:


gbm_coefficient_nonezero=gbm_coefficient[(gbm_coefficient.iloc[:,2]!=0)].iloc[:,0]
# print(len(gbm_coefficient_nonezero))
# gbm_coefficient_nonezero


# In[434]:


print(len(set(gbm_coefficient_nonezero.str.split('.',expand=True)[0])))
set(gbm_coefficient_nonezero.str.split('.',expand=True)[0])


# In[435]:


perf_gbm=best_gbm1.model_performance(train=True)  # Train ROC AUC 0.9543
perf_gbm_valid =best_gbm1.model_performance(valid)  # Valid ROC AUC 0.7271

perf_gbm.plot()
perf_gbm_valid.plot()

#### AUC_KS
AUC_KS(traindata['FUND_CNT'].as_data_frame(),best_gbm1.predict(traindata).as_data_frame()['p1'])
AUC_KS(valid['FUND_CNT'].as_data_frame(),best_gbm1.predict(valid).as_data_frame()['p1'])


# In[436]:


best_gbm1.auc()


# #### 201804

# In[437]:


pred_201804=best_gbm1.predict(h2o_df_P_201804).as_data_frame()

AUC_KS(h2o_df_P_201804_df['FUND_CNT'],pred_201804[['p1']])

data_201804=pd.concat([h2o_df_P_201804_df['FUND_CNT'],pred_201804[['p1']]],axis=1) 

MAPE(data_201804)


# In[438]:


Margin(pd.concat([h2o_df_P_201804_df['FUND_CNT'],pred_201804[['p1']],h2o_df_P_201804_df['NEWMARGIN']],axis=1) )


# #### 201805

# In[439]:


pred_201805=best_gbm1.predict(h2o_df_P_201805).as_data_frame()

AUC_KS(h2o_df_P_201805_df['FUND_CNT'],pred_201805[['p1']])

data_201805=pd.concat([h2o_df_P_201805_df['FUND_CNT'],pred_201805[['p1']]],axis=1) 

MAPE(data_201805)


# In[440]:


Margin(pd.concat([h2o_df_P_201805_df['FUND_CNT'],pred_201805[['p1']],h2o_df_P_201805_df['NEWMARGIN']],axis=1) )


# #### 201806

# In[441]:


pred_201806=best_gbm1.predict(h2o_df_P_201806).as_data_frame()

AUC_KS(h2o_df_P_201806_df['FUND_CNT'],pred_201806[['p1']])

data_201806=pd.concat([h2o_df_P_201806_df['FUND_CNT'],pred_201806[['p1']]],axis=1) 

MAPE(data_201806)


# In[442]:


Margin(pd.concat([h2o_df_P_201806_df['FUND_CNT'],pred_201806[['p1']],h2o_df_P_201806_df['NEWMARGIN']],axis=1) )


# #### 201807

# In[443]:


pred_201807=best_gbm1.predict(h2o_df_P_201807).as_data_frame()

AUC_KS(h2o_df_P_201807_df['FUND_CNT'],pred_201807[['p1']])

data_201807=pd.concat([h2o_df_P_201807_df['FUND_CNT'],pred_201807['p1']],axis=1) 

MAPE(data_201807)


# In[444]:


Margin(pd.concat([h2o_df_P_201807_df['FUND_CNT'],pred_201807[['p1']],h2o_df_P_201807_df['NEWMARGIN']],axis=1) )


# In[82]:


model_path = h2o.save_model(model=best_gbm1, force=True)
print(model_path)


# ### Full dataset

# In[98]:


aaa=best_gbm1.predict(h2o_DnB_F_model)


# In[106]:


bbb=pd.concat([aaa.as_data_frame(),DnB_F1],axis=1)
bbb.to_csv('H:/1_Xiaoxi_Ma_Project/Model C/Dataset/csv/GBM/all1.csv')


# In[ ]:


import sys
sys.path.insert(1,"../../")

import math


def partial_plot_test():
    kwargs = {}
    kwargs['server'] = True



    # Plot Partial Dependence for one feature then for both
    pdp1=best_gbm1.partial_plot(data=data,cols=['AGE'],server=True, plot=True)
    #Manual test
    h2o_mean_response_pdp1 = pdp1[0]["mean_response"]
    h2o_stddev_response_pdp1 = pdp1[0]["stddev_response"]
    h2o_std_error_mean_response_pdp1 = pdp1[0]["std_error_mean_response"]
    pdp_manual = partial_dependence(best_gbm1, data, "AGE", pdp1, 0)

    assert h2o_mean_response_pdp1 == pdp_manual[0]
    assert h2o_stddev_response_pdp1 == pdp_manual[1]
    assert h2o_std_error_mean_response_pdp1 == pdp_manual[2]

    pdp2=best_gbm1.partial_plot(data=data,cols=['AGE'    ,'RACE'], server=True, plot=False)
    #Manual test
    h2o_mean_response_pdp2 = pdp2[0]["mean_response"]
    h2o_stddev_response_pdp2 = pdp2[0]["stddev_response"]
    h2o_std_error_mean_response_pdp2 = pdp2[0]["std_error_mean_response"]
    pdp_manual = partial_dependence(best_gbm1, data, "AGE", pdp2, 0)

    assert h2o_mean_response_pdp2 == pdp_manual[0]
    assert h2o_stddev_response_pdp2 == pdp_manual[1]
    assert h2o_std_error_mean_response_pdp2 == pdp_manual[2]

    #Manual test
    h2o_mean_response_pdp2_race = pdp2[1]["mean_response"]
    h2o_stddev_response_pdp2_race = pdp2[1]["stddev_response"]
    h2o_std_error_mean_response_pdp2_race = pdp2[1]["std_error_mean_response"]
    pdp_manual = partial_dependence(best_gbm1, data, "RACE", pdp2, 1)

    assert h2o_mean_response_pdp2_race == pdp_manual[0]
    assert h2o_stddev_response_pdp2_race == pdp_manual[1]
    assert h2o_std_error_mean_response_pdp2_race == pdp_manual[2]

def partial_dependence(object, pred_data, xname, h2o_pp, pdp_name_idx):
    x_pt = h2o_pp[pdp_name_idx][xname.lower()] #Needs to be lower case here as the PDP response sets everything to lower
    y_pt = list(range(len(x_pt)))
    y_sd = list(range(len(x_pt)))
    y_sem = list(range(len(x_pt)))

    for i in range(len(x_pt)):
        x_data = pred_data
        x_data[xname] = x_pt[i]
        pred = object.predict(x_data)["p1"]
        y_pt[i] = pred.mean()[0,0]
        y_sd[i] = pred.sd()[0]
        y_sem[i] = y_sd[i]/math.sqrt(x_data.nrows)

    return y_pt, y_sd, y_sem

if __name__ == "__main__":
    pyunit_utils.standalone_test(partial_plot_test)
else:
    partial_plot_test()

