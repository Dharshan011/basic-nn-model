# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:V.dharshan
### Register Number:212222230031
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd


auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet=gc.open("DL").sheet1
data=worksheet.get_all_values()

dataset1=pd.DataFrame(data[1:],columns=data[0])
dataset1=dataset1.astype({'Input':'float'})
dataset1=dataset1.astype({'Output':'float'})

dataset1.head()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
X = dataset1[['Input']].values
y = dataset1[['Output']].values
X
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
MinMaxScaler()
X_train1 = Scaler.transform(X_train)
ai_brain=Sequential([Dense(units=1,input_shape=[1]),])
ai_brain.summary()
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(X_train1,y_train,epochs=2000)
loss_df=pd.DataFrame(ai_brain.history.history)
loss_df.plot()
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(X_train1,y_train,epochs=3000)

loss= pd.DataFrame(model.history.history)
loss.plot()

X_test1 =Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)

X_n1=[[4]]
X_n1_1=Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)
```
## Dataset Information
![d](https://github.com/Navyavenkat/basic-nn-model/assets/94165327/7d964891-a4a4-4511-a023-c85dc10d1435)


## OUTPUT

### Training Loss Vs Iteration Plot
![DL](https://github.com/Navyavenkat/basic-nn-model/assets/94165327/f7cb7f5f-87ba-4332-ad0f-7fa2a65f9a11)


### Test Data Root Mean Squared Error

![DL1](https://github.com/Navyavenkat/basic-nn-model/assets/94165327/fa7c560e-d210-477f-90fc-85873305018c)


### New Sample Data Prediction
![I](https://github.com/Navyavenkat/basic-nn-model/assets/94165327/5721ead5-b41b-4018-8248-4e4b07f1db29)


## RESULT

Hence the  development of  a neural network regression model for the given dataset is successfully developed

