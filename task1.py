import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_poisson_deviance
from sklearn.metrics import mean_gamma_deviance
from sklearn.metrics import mean_tweedie_deviance

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']
target = 'SalePrice'


train = train[features + [target]].dropna()

X = train[features]
y = np.log(train[target])  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

print(f'Training RMSE: {train_rmse}')
print(f'Testing RMSE: {test_rmse}')


test_predictions = np.exp(model.predict(test[features]))

submission_df = pd.DataFrame({'Id': test['Id'], 'SalePrice': test_predictions})
submission_df.to_csv('t1submission.csv', index=False)