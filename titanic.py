# start ipython notebook

import pandas as pd
 

input_df = pd.read_csv('train.csv', header=0)
submit_df  = pd.read_csv('test.csv',  header=0)
 

df = pd.concat([input_df, submit_df])
 

df.reset_index(inplace=True)
 

df.drop('index', axis=1, inplace=True)
df = df.reindex_axis(input_df.columns, axis=1)
 
print df.shape[1], "columns:", df.columns.values
print "Row count:", df.shape[0]






df['Cabin'][df.Cabin.isnull()] = 'U0'
df.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

#Use a regression or another simple model to predict the values of missing variables

def setMissingAges(df):
    
    age_df = df[['Age','Embarked','Fare', 'Parch', 'SibSp', 'Title_id','Pclass','Names','CabinLetter']]
    
    knownAge = age_df.loc[ (df.Age.notnull()) ]
    unknownAge = age_df.loc[ (df.Age.isnull()) ]
    
    y = knownAge.values[:, 0]
    X = knownAge.values[:, 1::]
    
    
    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rtr.fit(X, y)
    
    
    predictedAges = rtr.predict(unknownAge.values[:, 1::])
    
    
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    
    return df








