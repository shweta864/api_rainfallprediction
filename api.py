from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import svm
from sklearn.linear_model import LinearRegression
from mlxtend.regressor import StackingCVRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score

app = Flask(__name__)

# Read the data
data = pd.read_csv("C:/Users/SWHETA SHUKLA/Desktop/rainfall_predicition-main/rainfall_predicition-main/rainfall in india 1901-2015.csv")

# Data Preparation
data = data.fillna(data.mean(numeric_only=True))

# Choose a specific region (for demonstration purposes)
group = data.groupby('SUBDIVISION')[['YEAR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']]
data = group.get_group('EAST RAJASTHAN')

# Reshape the data
df = data.melt(['YEAR']).reset_index()
df = df[['YEAR','variable','value']].reset_index().sort_values(by=['YEAR','index'])
df.columns = ['Index','Year','Month','Avg_Rainfall']
df.drop(columns="Index", inplace=True)

# Mapping month names to numbers
Month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
df['Month'] = df['Month'].map(Month_map)

# Splitting the data into features (X) and target variable (y)
X = np.asanyarray(df[['Year', 'Month']]).astype('int')
y = np.asanyarray(df['Avg_Rainfall']).astype('int')

# Splitting the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# Linear Regression Model
LR = LinearRegression()
LR.fit(X_train, y_train)

# Random Forest Model
random_forest_model = RandomForestRegressor(max_depth=100, max_features='sqrt', min_samples_leaf=4,
                                             min_samples_split=10, n_estimators=800)
random_forest_model.fit(X_train, y_train)

# XGBoost Model
xgb = XGBRegressor()
xgb.fit(X_train, y_train)

# SVM Model
svm_regr = svm.SVC(kernel='rbf')
svm_regr.fit(X_train, y_train)

# Stacked Model
stack5 = StackingCVRegressor(regressors=(LR, xgb, svm_regr),
                             meta_regressor=random_forest_model, 
                             use_features_in_secondary=True,
                             store_train_meta_features=True,
                             shuffle=False,
                             random_state=42)
stack5.fit(X_train, y_train)

@app.route('/predict_rainfall', methods=['POST'])
def predict_rainfall():
    try:
        data = request.get_json(force=True)
        year_input = int(data['year'])
        month_input = int(data['month'])

        user_input = [[year_input, month_input]]

        # Predict using the stacked model
        predicted = stack5.predict(user_input)

        result = {
            'year': year_input,
            'month': month_input,
            'predicted_rainfall': predicted[0]
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000)
