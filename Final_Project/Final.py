import pandas as pd
df1=pd.read_csv('ACFTREF.csv')
df1.head()
df1.columns[0]
df2=pd.read_csv('MASTER.csv')
df2.head()
df2.columns[2]
import sqlite3
#!rm 'aircrafts.db'
open('aircrafts.db','w').close()
db = sqlite3.connect('aircrafts.db')
conn = db.cursor()
conn.execute("""CREATE TABLE codes (
                'N' text, 'CODE' text)""")
conn.execute("""CREATE TABLE models (
                'CODE' text, 'MFR' text, 'MODEL' text)""")

for i in df2.index:
    n = df2.ix[i,'\xef\xbb\xbfN-NUMBER']
    code = df2.ix[i,'MFR MDL CODE']
    conn.execute('INSERT INTO codes VALUES (?,?)', (n, code))
db.commit()

for i in df1.index:
    code = df1.ix[i,'\xef\xbb\xbfCODE']
    mfr = df1.ix[i,'MFR']
    model = df1.ix[i,'MODEL']
    conn.execute('INSERT INTO models VALUES (?,?,?)', (code,mfr,model))
db.commit()

conn.execute('SELECT codes.*, models.* FROM codes INNER JOIN models ON codes.CODE=models.CODE')
results=conn.fetchall()

df_res = pd.DataFrame(results)
df_res=df_res.drop(1, axis=1)
df_res=df_res.drop(2, axis=1)
df_res.head()

df_res.columns=['CODE','MFR', 'MODEL']
df_res.head()
df_res.to_csv('Aircrafts.csv')

import numpy as np
df0=pd.read_csv('SFO_JFK.csv')
import random
k=random.sample(df0.index, 1000)
df= df0.ix[k]
df=df.drop("Unnamed: 0",1)
df=df.drop("YEAR",1)
df=df.drop("DAY_OF_MONTH",1)
#df=df.drop("FL_DATE",1)
#df=df.drop("TAIL_NUM",1)
#df=df.drop("ORIGIN_AIRPORT_ID",1)
df=df.drop("ORIGIN_CITY_MARKET_ID",1)
df=df.drop("FLIGHTS",1)
#df=df.drop("DISTANCE",1)
df=df.drop("DISTANCE_GROUP",1)
#df=df.drop("FL_NUM",1)
df=df.drop("DEST_AIRPORT_ID",1)
#df=df.drop("CRS_DEP_TIME",1)
#df=df.drop("DEP_TIME",1)
#df=df.drop("DEP_DELAY",1)
df=df.drop("DEP_DEL15",1)
df=df.drop("DEP_DELAY_GROUP",1)
df=df.drop("TAXI_OUT",1)
df=df.drop("WHEELS_OFF",1)
df=df.drop("WHEELS_ON",1)
df=df.drop("TAXI_IN",1)
#df=df1.drop("CRS_ARR_TIME",1)
#df=df1.drop("ARR_TIME",1)
#df=df1.drop("ARR_DELAY",1)
#d1=df1.drop("ARR_DELAY_NEW",1)
df=df.drop("ARR_DEL15",1)
df=df.drop("ARR_DELAY_GROUP",1)
df=df.drop("ARR_TIME_BLK",1)
#df1=df1.drop("CANCELLED",1)
#df1=df1.drop("DIVERTED",1)
#df1=df1.drop("CRS_ELAPSED_TIME",1)
#df1=df1.drop("ACTUAL_ELAPSED_TIME",1)
#df1=df1.drop("AIR_TIME",1)
df=df.drop("Unnamed: 42",1)
df=df.drop("DEP_DELAY_NEW",1)
df['FL_NUM']=df.FL_NUM.map(str)+df['UNIQUE_CARRIER']

dfa=pd.read_csv('Aircrafts.csv')
dfa.head()
# Removing the first character in TAIL_NUM feature
df=df[df.TAIL_NUM.notnull()]
for i in df.index:
    a=df.ix[i,'TAIL_NUM']
    df.ix[i,'TAIL_NUM']=a[1:]
dfa.head()


df_cols=df.columns.tolist()
print df_cols, len(df_cols)
get_ipython().system(u"rm 'main.db'")

open('main.db','w').close()
db = sqlite3.connect('main.db')
conn = db.cursor()
conn.execute("""CREATE TABLE flights (
                'MONTH' text, 'DAY_OF_WEEK' text, 'FL_DATE' text, 
                'UNIQUE_CARRIER' text, 'TAIL_NUM' text, 'FL_NUM' text, 
                'ORIGIN_AIRPORT_ID' text, 'CRS_DEP_TIME' int, 'DEP_TIME' int, 
                'DEP_DELAY'int , 'DEP_TIME_BLK' text, 'CRS_ARR_TIME' int, 
                'ARR_TIME' int, 'ARR_DELAY' int, 'ARR_DELAY_NEW' int, 
                'CANCELLED' int, 'DIVERTED' int, 'CRS_ELAPSED_TIME' int, 
                'ACTUAL_ELAPSED_TIME' int, 'AIR_TIME' int, 'DISTANCE' int, 
                'CARRIER_DELAY' int, 'WEATHER_DELAY' int, 'NAS_DELAY' int, 
                'SECURITY_DELAY' int, 'LATE_AIRCRAFT_DELAY' int, 'Route text')""")

conn.execute("""CREATE TABLE aircrafts (TAIL_NUM text, MFR text, MODEL text)""")

for i in df.index:
    each_entry=[]
    for j in df_cols:
        a = df.ix[i,j]
        each_entry.append(a)
    conn.execute('INSERT INTO flights VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', (each_entry))
db.commit()

for i in dfa.index:
    tail = dfa.ix[i,'CODE']
    mfr = dfa.ix[i,'MFR']
    model = dfa.ix[i,'MODEL']
    conn.execute('INSERT INTO aircrafts VALUES (?,?,?)', (tail, mfr,model))
db.commit()

conn.execute('SELECT flights.*, aircrafts.* FROM flights INNER JOIN aircrafts ON flights.TAIL_NUM=aircrafts.TAIL_NUM')
results=conn.fetchall()

df_cols.append('TAIL_NUM_1')
df_cols.append('MFR')
df_cols.append('MODEL')
print df_cols, len(df_cols)



df_res = pd.DataFrame(results)
df_res.columns=df_cols
df_res=df_res.drop('TAIL_NUM_1', axis=1)
df_res.head()


# Changing the order of the columns (features to the left, outputs to the right)

def set_column_sequence(dataframe, seq):
    cols = seq[:]
    for x in dataframe.columns:
        if x not in cols:
            cols.append(x)
    return dataframe[cols]

seq=['MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'UNIQUE_CARRIER', 'TAIL_NUM', 'MFR', 'MODEL', 'DEP_TIME', 'DEP_TIME_BLK', 'ARR_TIME', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'ROUTE'] 
df_res=set_column_sequence(df_res,seq)
df_res.head()

df_res.to_csv('cleanup1.csv')

df=pd.read_csv('cleanup1.csv')
df=df.drop('Unnamed: 0', axis=1)
df=df.drop('TAIL_NUM', axis=1)
df=df.drop('AIR_TIME', axis=1)
df=df.drop('FL_DATE', axis=1)

df['ACT_ARR_TIME']=df['ARR_TIME']
df['ACT_DEP_TIME']=df['DEP_TIME']
df=df[df.CRS_ARR_TIME.notnull()]
df=df[df.CRS_DEP_TIME.notnull()]
df=df[df.ACT_ARR_TIME.notnull()]
df=df[df.ACT_DEP_TIME.notnull()]
df.head()


# All this formatting

cols_format=['ACT_DEP_TIME', 'ACT_ARR_TIME', 'CRS_DEP_TIME', 'CRS_ARR_TIME']
for i in df.index:
    for j in cols_format:
        k=round(df.ix[i,j])
        k=int(k)
        df.ix[i,j]=str(k)
df.head()


for i in cols_format:
    for j in range(1,10):
        df = df.drop(df.index[df[i] == str(j)])
    df = df.drop(df.index[df[i] == '2400'])
df.head()


# Formatting the dates

from datetime import datetime, date

FMT='%H%M'
FMT1='%H:%M'


for i in df.index:
    
    df.ix[i,'ACT_DEP_TIME']=datetime.strptime(df.ix[i,'ACT_DEP_TIME'], FMT).strftime(FMT1)
    df.ix[i,'ACT_ARR_TIME']=datetime.strptime(df.ix[i,'ACT_ARR_TIME'], FMT).strftime(FMT1)
    df.ix[i,'CRS_DEP_TIME']=datetime.strptime(df.ix[i,'CRS_DEP_TIME'], FMT).strftime(FMT1)
    df.ix[i,'CRS_ARR_TIME']=datetime.strptime(df.ix[i,'CRS_ARR_TIME'], FMT).strftime(FMT1)
    
    crs_dur = datetime.strptime(df.ix[i,'CRS_ARR_TIME'], FMT1) - datetime.strptime(df.ix[i,'CRS_DEP_TIME'], FMT1)
    df.ix[i,'CRS_DURATION']=crs_dur.total_seconds()
    act_dur = datetime.strptime(df.ix[i,'ACT_ARR_TIME'], FMT1) - datetime.strptime(df.ix[i,'ACT_DEP_TIME'], FMT1)
    df.ix[i,'ACT_DURATION']=act_dur.total_seconds()

df.head()


print df.columns.tolist()
cols_features=['MONTH', 'DAY_OF_WEEK', 'MFR', 'MODEL', 'DEP_TIME_BLK',  'FL_NUM']


N_pred=len(cols_features)
print N_pred


df[cols_features].head()


# In[14]:

d_cols=['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']
d_df=df[d_cols]
for i in df.index:
    for j in d_df.columns:
        j1=str(j)+str('_01')
        if (df.ix[i,j]==0):
            df.ix[i,j1]=0
        else:
            df.ix[i,j1]=1


df=df.fillna(0)
df.head()


for i in df.index:
    if df.ix[i,'CRS_DURATION']<0:
        df.ix[i,'CRS_DURATION']=df.ix[i,'CRS_DURATION']+86400
    df.ix[i,'CRS_DURATION']=df.ix[i,'CRS_DURATION']/60
df.head()

for i in df.index:
    if df.ix[i,'ACT_DURATION']<0:
        df.ix[i,'ACT_DURATION']=df.ix[i,'ACT_DURATION']+86400
    df.ix[i,'ACT_DURATION']=df.ix[i,'ACT_DURATION']/60
df.ix[:, 10:]


df.to_csv('cleaned.csv')

df=pd.read_csv('cleaned.csv')
df=df.drop("Unnamed: 0",1)
df=df.drop('ACT_DURATION', axis=1)
df=df.drop('CRS_DURATION', axis=1)
df=df.drop('CRS_DEP_TIME', axis=1)
df=df.drop('CRS_ARR_TIME', axis=1)
df=df.drop('ACT_ARR_TIME', axis=1)
df=df.drop('ACT_DEP_TIME', axis=1)
df=df.drop('DISTANCE', axis=1)



f_cols=['MONTH', 'DAY_OF_WEEK', 'MFR', 'MODEL', 'DEP_TIME_BLK', 'FL_NUM','CRS_ELAPSED_TIME']


#df1.groupby('UNIQUE_CARRIER').UNIQUE_CARRIER.count().plot(kind='bar', color='red')
#grouped=df1.groupby('MONTH')
#print grouped.DELAY.mean()

# Exploring unique values

f_df=df[f_cols]
for i in f_df.columns:
    values = df[i].unique()
    values.sort()
print "For {0} we have: {1} as possible values".format(i, values)
print


df['MONTH']=df['MONTH'].replace({1: 'JAN', 2: 'FEB',3: 'MAR',4: 'APR',5: 'MAY',6: 'JUN',7: 'JUL',8: 'AUG',9: 'SEP',10: 'OCT',11: 'NOV',12: 'DEC'})
df['DAY_OF_WEEK']=df['DAY_OF_WEEK'].replace({1: 'Mon', 2: 'Tue',3: 'Wed',4: 'Thu',5: 'Fri',6: 'Sat',7: 'Sun'})
df.head()
df['MFR']=df['MFR'].replace({'AIRBUS INDUSTRIE              ':'AIRBUS                        '})


####### The length of the trip



#df1['ACTUAL_ELAPSED_TIME'] = (df1['ACTUAL_ELAPSED_TIME']-df1['ACTUAL_ELAPSED_TIME'].mean())/df1['ACTUAL_ELAPSED_TIME'].std()


# Function for creating dummies for categorical variables

for i in f_df.columns[:-1]:
    d_table=pd.get_dummies(df[i])
    df=df.join(d_table)


df=df.fillna(0)


df=df.dropna()
df['ELAPSED_TIME_L']=df['CRS_ELAPSED_TIME']
x = df.ix[:, 28:]
y = df['ACTUAL_ELAPSED_TIME']

print x



import statsmodels.api as sm

X = sm.add_constant(x, prepend=True)
results = sm.OLS(y, X).fit()



results.summary()



from sklearn.linear_model import LinearRegression
model_lin=LinearRegression(fit_intercept=True, normalize=False, copy_X=True)
model_lin.fit(x,y)
df['predicted duration']=model_lin.predict(x)
df[['predicted duration', 'ACTUAL_ELAPSED_TIME','CRS_ELAPSED_TIME']].head()


####### what is better: my prediction or the scheduled time?


x1 = df['CRS_ELAPSED_TIME']
y1 = df['ACTUAL_ELAPSED_TIME']
X1 = sm.add_constant(x1, prepend=True)
results = sm.OLS(y1, X1).fit()


results.summary()



