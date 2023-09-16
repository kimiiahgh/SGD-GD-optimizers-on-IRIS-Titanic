import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import  Dense
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from numpy import nan
np.random.seed(1)
######iris
iris = datasets.load_iris()
x=iris.data
y=iris.target.reshape(-1, 1)
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.5)
def build_model_iris(n):
  model=keras.Sequential()
  for i in range(0,n):
    model.add(Dense(10,activation=tf.keras.activations.linear))
  model.add(Dense(3,activation=tf.nn.softmax))
  #model.add(Dense(3,activation=tf.keras.activations.linear))  
  model.compile(loss='mse')
  return model
T_err=[]  
for i in range(0,3):
  mse=[0,0]
  model=build_model_iris(i)
  model.fit(train_x,train_y,epochs=1000)
  mse_tr=model.evaluate(train_x,train_y)
  mse_te=model.evaluate(test_x,test_y)
  mse[0]=mse_tr
  mse[1]=mse_te
  T_err.append(mse)
for i in range(0,3):
  print("MSE error for a NN on iris with",i,"hidden layer on train data is:",T_err[i][0])
  print("MSE error for a NN on iris with",i,"hidden layer on test data is:",T_err[i][1])   
#######titanic
#import io
#df = pd.read_csv(io.BytesIO(uploaded['titanic.csv']))
df=pd.read_csv('titanic.csv')
df[df.columns]=df[df.columns].replace('?',nan)
#preproccesing
df['age']=pd.to_numeric(df['age'], downcast='float')
df['fare']=pd.to_numeric(df['fare'],downcast='float')
#df['embarked'].to_string(df['embarked'])
df['age']=df.groupby(['pclass'])['age'].transform(lambda x: x.fillna(x.mean()))
df['fare'].fillna(df["fare"].mean(), inplace=True)
df['embarked'].fillna('S',inplace=True)
#drop columns
data=df.drop(['name','ticket','cabin','boat','body','home.dest'],axis=1)
#transform sex to  0,1
labelencoder_X = LabelEncoder() 
data['sex'] = labelencoder_X.fit_transform(data['sex']) #0-> female 1->male
#transform embarked to 0,1,2
data['embarked']=labelencoder_X.fit_transform(data['embarked'])
y=data['survived']
x=data.drop('survived',axis=1)
np.random.seed(1)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.5)
def build_model_titanic(n):
  model=keras.Sequential()
  for i in range(0,n):
    model.add(Dense(10,activation=tf.keras.activations.linear))
  model.add(Dense(1,activation=tf.keras.activations.linear))
  #model.add(Dense(1,activation=tf.keras.activations.linear))  
  model.compile(loss='mse')
  return model
T_err=[]  
for i in range(0,3):
  mse=[0,0]
  model=build_model_titanic(i)
  model.fit(train_x,train_y,epochs=1000)
  mse_tr=model.evaluate(train_x,train_y)
  mse_te=model.evaluate(test_x,test_y)
  mse[0]=mse_tr
  mse[1]=mse_te
  T_err.append(mse)
  #rmse=np.sqrt(mse)
for i in range(0,3):
  print("MSE error for a NN on titanic with",i,"hidden layer on train data is:",T_err[i][0])
  print("MSE error for a NN on titanic with",i,"hidden layer on test data is:",T_err[i][1]) 





