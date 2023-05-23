# Import necessary libraries
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from pandas import DataFrame, concat
import numpy as np
from polygon import RESTClient
import pandas as pd

# create client
c = RESTClient(api_key="u5Blk2NMTBn6mecFlLqutoNNsPrJrq4W")

# list of tickers you're interested in
ticker = "SPY"

# loop through each ticker
bars = c.get_aggs(ticker=ticker, multiplier=1, timespan="day", from_="2021-05-22", to="2023-05-22")

bars_df = pd.DataFrame([bar.__dict__ for bar in bars])

bars_df = bars_df.drop("otc", axis=1)

# Create lagged features
for i in range(1, 11):
    bars_df[f'lag_{i}'] = bars_df['close'].shift(i)

# imp = IterativeImputer(max_iter=10, random_state=42)
# imp.fit(bars_df[11:])

# the model learns that the second feature is double the first
# imputed = pd.DataFrame(imp.transform(bars_df[:11]))
#
# bars_df[:11] = imputed

bars_df['profitable'] = bars_df['close'] > bars_df['open']
# Create rolling window features
for window in [7, 14, 30]:
    bars_df[f'rolling_mean_{window}'] = bars_df['close'].rolling(window).mean()
    bars_df[f'rolling_std_{window}'] = bars_df['close'].rolling(window).std()


# Drop missing values
bars_df = bars_df.dropna()

# Split data into features (X) and target (y)
X = bars_df.drop(['profitable', 'close'], axis=1)
y = bars_df['profitable']

# Normalize the data
sc = StandardScaler()
X = sc.fit_transform(X)

# Split the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)

# Fit the model to the training data
clf.fit(X_train, y_train)

# Use the model to predict the test set
y_pred = clf.predict(X_test)


# Print the accuracy of the model
print("Accuracy: ", clf.score(X_test, y_test))
