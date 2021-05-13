import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
 

houseDF = pd.read_csv(r'C:\Users\Lenovo\Desktop\datasets\data.csv')
houseDF.head()
houseDF.columns()

#  Index(['date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
#        'floors', 'waterfront', 'view', 'condition', 'sqft_above',
#        'sqft_basement', 'yr_built', 'yr_renovated', 'street', 'city',
#        'statezip', 'country'],
#         dtype='object')

X = houseDF[['date', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
       'floors', 'waterfront', 'view', 'condition', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated', 'street', 'city',
       'statezip', 'country']]
y = houseDF['price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from sklearn.linear_model import LinearRegression
lm = LinearRegression() # creating a linear regression object
lm.fit(X_train,y_train) # fitting a model inside the object
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df

prediction = lm.predict(X_test)
plt.scatter(y_test, prediction)
