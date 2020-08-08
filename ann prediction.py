#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.callbacks import History
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics


history = History()
df=pd.read_csv('/home/simriti/Desktop/final1.csv')

df=df.dropna()
a=list(df.columns)
X = df[[a[1],a[2],a[3], a[4]]]
Y = df[a[5]]



df2=pd.read_csv('/home/simriti/Desktop/final2.csv')

b=list(df2.columns)
X2 = df2[[b[1],b[2],b[3], b[4]]]
Y2 = df2[b[5]]


model = Sequential()

model.add(Dense(1000, activation='sigmoid', input_shape=(4,)))
model.add(Dense(990, activation='sigmoid'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
print(model.summary())

X_train = X
Y_train = Y

#X_test = X2
#Y_test = Y2

X_train, X_test, Y_train, Y_test = train_test_split(X ,Y, test_size=0.9, random_state=42)
history = model.fit(X_train, Y_train,validation_data=(X_test,Y_test),epochs=100,batch_size=100,verbose=2, callbacks = [history])
#print(X_test)
#print(Y_test)

predictions = model.predict(X2)
print(predictions)

#print(history.history.keys())


model.save('/home/simriti/Desktop/model_trial.h5')
print("Saved model to disk")


plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])

plt.ylabel('Mean Squared Error')
plt.xlabel('epoch')
plt.legend(['train','test'])
plt.show()

plt.plot(Y2,label="Actual")
plt.plot(predictions, label="predicted")
#plt.plot(history.history['val_mse'])

plt.ylabel('gwl')
plt.xlabel('Test Input')
plt.legend(['train','test'])
plt.show()


# In[ ]:





# In[ ]:




