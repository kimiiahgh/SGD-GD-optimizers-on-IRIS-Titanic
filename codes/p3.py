import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import  Dense
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder
np.random.seed(1)
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from numpy import nan
import matplotlib.pyplot as plt

#model 
#opt 
def SGD(x,y,i,size): #hanuz motmaen nistam!
  model=keras.Sequential()
  model.add(Dense(10,activation=tf.keras.activations.sigmoid))
  if i==1:
    model.add(Dense(3,activation=tf.nn.softmax)) #iris
  elif i==2:
    model.add(Dense(1,activation=tf.keras.activations.sigmoid))  #titanic
  model.compile(optimizer='sgd',loss='mse')
  history=model.fit(x,y,batch_size=size,epochs=1000)
  hist = pd.DataFrame(history.history)
  hist.head()
  hist['epoch'] = history.epoch
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('loss')
  plt.plot(hist['epoch'], history.history['loss'], label='Train Error')
  plt.legend()
  plt.ylim([0,(model.evaluate(x,y)*(i+1))])
  return model 
def GD(x,y,i):
  size=len(x)
  model=keras.Sequential()
  model.add(Dense(10,activation=tf.keras.activations.sigmoid))
  if i==1:
    model.add(Dense(3,activation=tf.nn.softmax)) #iris
  elif i==2:
    model.add(Dense(1,activation=tf.keras.activations.sigmoid)) #titanic
  model.compile(optimizer='sgd',loss='mse')
  history=model.fit(x,y,batch_size=size,epochs=1000)
  hist = pd.DataFrame(history.history)
  hist.head()
  hist['epoch'] = history.epoch
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('loss')
  plt.plot(hist['epoch'], history.history['loss'], label='Train Error')
  plt.legend()
  plt.ylim([0,(model.evaluate(x,y)*(i+1))])
  return model
  return model
def Adagrad(x,y,i): 
  model=keras.Sequential()
  model.add(Dense(10,activation=tf.keras.activations.sigmoid))
  if i==1:
    model.add(Dense(3,activation=tf.nn.softmax)) #iris
  elif i==2:
    model.add(Dense(1,activation=tf.keras.activations.sigmoid)) #titanic
  model.compile(optimizer='adagrad',loss='mse')  
  model.fit(x,y,epochs=1000)
  return model
def Adam(x,y,i):
  model=keras.Sequential()
  model.add(Dense(10,activation=tf.keras.activations.sigmoid))
  if i==1:
    model.add(Dense(3,activation=tf.nn.softmax)) #iris
  elif i==2:
    model.add(Dense(1,activation=tf.keras.activations.sigmoid)) #titanic 
  model.compile(optimizer='adam',loss='mse')  
  model.fit(x,y,epochs=1000)
  return model 
def RMSprop(x,y,i):
  opt=tf.keras.optimizers.RMSprop()
  model=keras.Sequential()
  model.add(Dense(10,activation=tf.keras.activations.sigmoid))
  if i==1:
    model.add(Dense(3,activation=tf.nn.softmax)) #iris
  elif i==2:
    model.add(Dense(1,activation=tf.keras.activations.sigmoid)) #titanic
  model.compile(opt,loss='mse')  
  model.fit(x,y,epochs=1000)
  return model 
def Momentum(x,y,i):
  opt=tf.keras.optimizers.SGD(momentum=0.9) #nemidunam daqiq chand bezaram!
  model=keras.Sequential()
  model.add(Dense(10,activation=tf.keras.activations.sigmoid))
  if i==1:
    model.add(Dense(3,activation=tf.nn.softmax)) #iris
  elif i==2:
    model.add(Dense(1,activation=tf.keras.activations.sigmoid)) #titanic
  model.compile(opt,loss='mse')  
  model.fit(x,y,epochs=1000)
  return model     
def Nestrov(x,y,i):
  opt=tf.keras.optimizers.SGD(nesterov=True)
  model=keras.Sequential()
  model.add(Dense(10,activation=tf.keras.activations.sigmoid))
  if i==1:
    model.add(Dense(3,activation=tf.nn.softmax)) #iris
  elif i==2:
    model.add(Dense(1,activation=tf.keras.activations.sigmoid)) #titanic
  model.compile(opt,loss='mse')  
  model.fit(x,y,epochs=1000)
  return model 
#iris
iris = datasets.load_iris()
x=iris.data
y=iris.target.reshape(-1, 1)
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.5)
#model
#SGD:
batches=[]
for i in range(0,9):
  batches.append(pow(2,i))
T_err=[]  
for i in batches:
  model=SGD(train_x,train_y,1,i)
  mse=[0,0] 
  mse_tr=model.evaluate(train_x,train_y)
  mse_te=model.evaluate(test_x,test_y)
  mse[0]=mse_tr
  mse[1]=mse_te
  T_err.append(mse)
for i in range(0,9):
    print("MSE error for a NN on iris train data and batch size=",batches[i],"is:",T_err[i][0])
    print("MSE error for a NN on iris test data and batch size=",batches[i],"is:",T_err[i][1])
#GD:
T_err=[] 
model=GD(train_x,train_y,1)
mse=[0,0] 
mse_tr=model.evaluate(train_x,train_y)
mse_te=model.evaluate(test_x,test_y)
mse[0]=mse_tr
mse[1]=mse_te
T_err.append(mse)
print("MSE error for a NN on iris train data using GD is:",T_err[0][0])
print("MSE error for a NN on iris test data using GD is:",T_err[0][1])
#Momentum:
T_err=[] 
model=Momentum(train_x,train_y,1)
mse=[0,0] 
mse_tr=model.evaluate(train_x,train_y)
mse_te=model.evaluate(test_x,test_y)
mse[0]=mse_tr
mse[1]=mse_te
T_err.append(mse)
print("MSE error for a NN on iris train data using Momentum is:",T_err[0][0])
print("MSE error for a NN on iris test data using Momentum is:",T_err[0][1])
#Nestrov:
T_err=[] 
model=Nestrov(train_x,train_y,1)
mse=[0,0] 
mse_tr=model.evaluate(train_x,train_y)
mse_te=model.evaluate(test_x,test_y)
mse[0]=mse_tr
mse[1]=mse_te
T_err.append(mse)
print("MSE error for a NN on iris train data using Nestrov is:",T_err[0][0])
print("MSE error for a NN on iris test data using Nestrov is:",T_err[0][1])
#RMSprop:
T_err=[] 
model=RMSprop(train_x,train_y,1)
mse=[0,0] 
mse_tr=model.evaluate(train_x,train_y)
mse_te=model.evaluate(test_x,test_y)
mse[0]=mse_tr
mse[1]=mse_te
T_err.append(mse)
print("MSE error for a NN on iris train data using RMSprop is:",T_err[0][0])
print("MSE error for a NN on iris test data using RMSprop is:",T_err[0][1])
#Adam:
T_err=[] 
model=Adam(train_x,train_y,1)
mse=[0,0] 
mse_tr=model.evaluate(train_x,train_y)
mse_te=model.evaluate(test_x,test_y)
mse[0]=mse_tr
mse[1]=mse_te
T_err.append(mse)
print("MSE error for a NN on iris train data using Adam is:",T_err[0][0])
print("MSE error for a NN on iris test data using Adam is:",T_err[0][1])
#Adagrad:
T_err=[] 
model=Adagrad(train_x,train_y,1)
mse=[0,0] 
mse_tr=model.evaluate(train_x,train_y)
mse_te=model.evaluate(test_x,test_y)
mse[0]=mse_tr
mse[1]=mse_te
T_err.append(mse)
print("MSE error for a NN on iris train data using Adagrad is:",T_err[0][0])
print("MSE error for a NN on iris test data using Adagrad is:",T_err[0][1])


###############titanic
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
#model
#SGD:
batches=[]
for i in range(0,9):
  batches.append(pow(2,i))
T_err=[]  
for i in batches:
  model=SGD(train_x,train_y,2,i)
  mse=[0,0] 
  mse_tr=model.evaluate(train_x,train_y)
  mse_te=model.evaluate(test_x,test_y)
  mse[0]=mse_tr
  mse[1]=mse_te
  T_err.append(mse)
for i in range(0,9):
    print("MSE error for a NN on titanic train data and batch size=",batches[i],"is:",T_err[i][0])
    print("MSE error for a NN on titanic test data and batch size=",batches[i],"is:",T_err[i][1])
#GD:
T_err=[] 
model=GD(train_x,train_y,2)
mse=[0,0] 
mse_tr=model.evaluate(train_x,train_y)
mse_te=model.evaluate(test_x,test_y)
mse[0]=mse_tr
mse[1]=mse_te
T_err.append(mse)   
print("MSE error for a NN on titanic train data using GD is:",T_err[0][0])
print("MSE error for a NN on titanic test data using GD is:",T_err[0][1])  
#Momentum:
T_err=[] 
model=Momentum(train_x,train_y,2)
mse=[0,0] 
mse_tr=model.evaluate(train_x,train_y)
mse_te=model.evaluate(test_x,test_y)
mse[0]=mse_tr
mse[1]=mse_te
T_err.append(mse)
print("MSE error for a NN on titanic train data using Momentum is:",T_err[0][0])
print("MSE error for a NN on titanic test data using Momentum is:",T_err[0][1])
#Nestrov:
T_err=[] 
model=Nestrov(train_x,train_y,2)
mse=[0,0] 
mse_tr=model.evaluate(train_x,train_y)
mse_te=model.evaluate(test_x,test_y)
mse[0]=mse_tr
mse[1]=mse_te
T_err.append(mse)
print("MSE error for a NN on titanic train data using Nestrov is:",T_err[0][0])
print("MSE error for a NN on titanic test data using Nestrov is:",T_err[0][1])
#RMSprop:
T_err=[] 
model=RMSprop(train_x,train_y,2)
mse=[0,0] 
mse_tr=model.evaluate(train_x,train_y)
mse_te=model.evaluate(test_x,test_y)
mse[0]=mse_tr
mse[1]=mse_te
T_err.append(mse)
print("MSE error for a NN on titanic train data using RMSprop is:",T_err[0][0])
print("MSE error for a NN on titanic test data using RMSprop is:",T_err[0][1])
#Adam:
T_err=[] 
model=Adam(train_x,train_y,2)
mse=[0,0] 
mse_tr=model.evaluate(train_x,train_y)
mse_te=model.evaluate(test_x,test_y)
mse[0]=mse_tr
mse[1]=mse_te
T_err.append(mse)
print("MSE error for a NN on titanic train data using Adam is:",T_err[0][0])
print("MSE error for a NN on titanic test data using Adam is:",T_err[0][1])
#Adagrad:
T_err=[] 
model=Adagrad(train_x,train_y,2)
mse=[0,0] 
mse_tr=model.evaluate(train_x,train_y)
mse_te=model.evaluate(test_x,test_y)
mse[0]=mse_tr
mse[1]=mse_te
T_err.append(mse)
print("MSE error for a NN on titanic train data using Adagrad is:",T_err[0][0])
print("MSE error for a NN on titanic test data using Adagrad is:",T_err[0][1])
