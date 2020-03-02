import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Reading data from the csv file

df = pd.read_csv("train_V2.csv")
test = pd.read_csv("test_V2.csv")





mType = pd.get_dummies(df['matchType'])
mTypeTest = pd.get_dummies(test['matchType'])
test=pd.concat([test,mTypeTest], axis=1)
df=pd.concat([df,mType], axis=1)
df=df.drop(['matchType'],axis=1)
test=test.drop(['matchType'],axis=1)
df=df.drop(['Id', 'groupId', 'matchId'],axis=1)
df.head()




df['damageDealt'].plot.hist()
plt.axis()
plt.xlabel('Damage Dealt')


test=test.drop(['Id', 'groupId', 'matchId'],axis=1)

from sklearn.model_selection import train_test_split 
X_train= df.drop(['winPlacePerc'],axis=1)
y_train= df['winPlacePerc']
X_test= test


#x=df
#y=test





from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)




from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x, y)

y_pred = regressor.predict(X_test)



y_pred.mean()

test2 = pd.read_csv("test2.csv")

# THIS GRAPH SHOWS DIFFERENT MATCH TYPE AND WINNING PERCENTAGE PREDICTION  

plt.figure(figsize=(12,4)) # this creates a figure 8 inch wide, 4 inch high
sns.barplot(x='matchType',y=y_pred,data=test2,ci=95)
plt.show()

y_pred.argmax()

print (y_pred[68]) 

# PRINT DATA OF PLAYER WITH MAXIMUM WINNING PERCENTAGE
test2[68:69]

test2[68:69]['matchType']

testfinal=test2






























