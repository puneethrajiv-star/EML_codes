#implement logestic regression algorithm for stock prices

import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


stock = yf.download("AAPL",start="2020-01-01",
                    end="2025-01-01")

print(f"type: {type(stock)}")
print(f"head: {stock.head()}")
print(f"columns: {stock.columns}")
print(f"no.of columns: {len(stock.columns)}")
print(f"info: {stock.info()}")

#---------------------------
# step 2: feature engneering
#---------------------------

stock['Returns']=stock['Close'].pct_change()
stock['sma_5']=stock['Close'].rolling(5).mean()
stock['sma_20']=stock['Close'].rolling(20).mean()
stock['sma_50']=stock['Close'].rolling(50).mean()
stock['Volatility']=stock['Returns'].rolling(20).std()

print(f"info: {stock.info()}")

stock.dropna(inplace=True)


#Target 1: if tommorrows price > todays price
stock['Target']=(stock['Close'].shift(-1)>stock['Close']).astype(int)
print(stock.head())

#prepare the data

features=['Close',
          'High',
          'Low',
          'Open',
          'Volume',
          'Returns',
          'sma_5',
          'sma_20',
          'sma_50',
          'Volatility']
features1=stock.columns
print(features1)

X = stock[features1]
X = X.drop('Target',axis=1)
y = stock['Target'].values

x_train,x_test,y_train,y_test = train_test_split( X, y,
                                                 test_size = 0.2,
                                                 random_state = 42)

# Build the model
model = LogisticRegression()

model.fit(x_train,y_train)
y_pred = model.predict(x_test)

print(f'accuracy : {accuracy_score(y_test, y_pred)}')
print(f'classification_report : {classification_report(y_test, y_pred)}')
print(f'confusion_matrix : {confusion_matrix(y_test, y_pred)}')

print(type(model.coef_[0]))

#visualization 

import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))
plt.bar(features, model.coef_[0])
plt.title("feature importance (logistic regression)")
plt.xlabel("features")
plt.ylabel("feature importance")
plt.savefig("barplot(feature importance)")
plt.show()

model=XGBClassifier(n_estimators=500,
                    learning_rate=0.1,
                    device='cuda'
                    )
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

print(f'accuracy : {accuracy_score(y_test, y_pred)}')
print(f'classification_report : {classification_report(y_test, y_pred)}')
print(f'confusion_matrix : {confusion_matrix(y_test, y_pred)}')
