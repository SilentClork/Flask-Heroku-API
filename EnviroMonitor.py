from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # 獲取 JSON 資料
    print(f"Received data: {data}")
    
    df = pd.DataFrame(data)  # 將資料轉換成 DataFrame
    print(f"DataFrame: {df.head()}")

    # 假設接收到的數據已經是正確的格式和維度
    X = df.drop('Random Score', axis=1)
    y = df['Random Score']

    # 創建模型字典
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'LinearRegression': LinearRegression(),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # 初始化 feature_importances
        feature_importances = np.zeros(X.shape[1])

        if name in ['RandomForest', 'GradientBoosting']:
            feature_importances = model.feature_importances_
        elif name == 'LinearRegression':
            feature_importances = np.abs(model.coef_)

        feature_importances = feature_importances / feature_importances.sum()  # 計算佔比

        # 找到最重要的特徵
        most_important_feature = X.columns[np.argmax(feature_importances)]

        results[name] = {
            'MSE': mse,
            'R2': r2,
            'Feature Importances': {feature: importance for feature, importance in zip(X.columns, feature_importances)},
            'Most Important Feature': most_important_feature
        }

        print(f"Response: {results}")

    # 返回 JSON 回應
    return jsonify(results)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)