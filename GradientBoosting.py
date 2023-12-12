import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor

import time

# Đọc dữ liệu từ file CSV
data = pd.read_csv('data_newv2.csv')
data.dropna()
data.drop_duplicates()
data.head(5)

# Chia thuộc tính "date" thành ngày, tháng và năm
data['date'] = pd.to_datetime(data['date'])
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['year'] = data['date'].dt.year

# Xóa cột "date" gốc
data = data.drop('date', axis=1)

# Chia thuộc tính "time" thành giờ, phút và giây
data['hour'] = data['time'].str.split(':').str[0]
data['minute'] = data['time'].str.split(':').str[1]
data['second'] = data['time'].str.split(':').str[2]

# Chuyển đổi giờ, phút và giây thành kiểu số nguyên
data['hour'] = data['hour'].astype(int)
data['minute'] = data['minute'].astype(int)
data['second'] = data['second'].astype(int)

# Xóa cột "time" gốc
data = data.drop('time', axis=1)

data = data[data['price'] > 0]
data['rooms'].unique()
data = data[data['rooms'] != -2]
data.loc[data['rooms'] == -1, 'rooms'] = 0
data['object_type'].unique()
data.loc[data['object_type'] == 11, 'object_type'] = 2
data = data[data.price.between(data.price.quantile(0.05), data.price.quantile(0.95))]
data = data[data.area.between(data.area.quantile(0.01), data.area.quantile(0.99))]

# Chia dữ liệu thành đặc trưng (X) và nhãn (y)
X = data[['month', 'day', 'year', 'hour', 'minute', 'second', 'geo_lat', 'geo_lon', 'region', 'building_type', 'level', 'levels', 'rooms', 'area', 'kitchen_area', 'object_type']]
y = data['price']

# Chuẩn hóa dữ liệu để sử dụng cho quá trình huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Hàm trực quan hóa
def visualize(y_test,y_pred):
    random.seed(42)  
    sample_indices = random.sample(range(len(y_test)), 50000)
    sample_y_test = y_test.iloc[sample_indices]
    sample_y_pred = y_pred[sample_indices]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(sample_y_test, sample_y_pred, c='blue', label='Actual vs. Predicted', alpha=0.7)
    plt.plot([min(sample_y_test), max(sample_y_test)], [min(sample_y_test), max(sample_y_test)], 'k--', lw=2, label='Ideal Line')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs. Predicted House Prices - GradientBoosting")
    plt.legend()
    plt.grid(True)
    fname = "GradientBoosting.png"
    plt.savefig(fname, dpi=100)
    plt.show()

#Hàm tính toán
def model_rp(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2):", r2)
    print("Mean Absolute Error(MAE):", mae)
    visualize(y_test,y_pred)

models= []

# Tạo mô hình hồi quy tuyến tính
GB_model = GradientBoostingRegressor()

# Đào tạo mô hình trên tập huấn luyện
GB_model.fit(X_train_scaled, y_train)

# Dự đoán giá nhà trên tập kiểm tra
GB_y_pred = GB_model.predict(X_test_scaled)

# In ra kết quả trực quan và tính toán
model_rp(y_test, GB_y_pred)
models.append(GB_model)

# Lưu mô hình tuyến tính thành một tệp .pkl
model_filename = 'GB_model.pkl'
joblib.dump(GB_model, model_filename)

# Tập dữ liệu của căn nhà mới 
new_house = [[2, 19, 2018, 20, 20, 21, 59.8058084, 30.376141, 2661, 1, 8, 10, 3, 82.6, 10.8, 1]]

# Chuẩn hóa dữ liệu căn nhà mới
new_house_scaled = scaler.transform(new_house)

# Dự đoán giá nhà cho căn nhà mới từ tập dữ liệu
new_price = GB_model.predict(new_house_scaled)

print(f"Predicted price for the new house: {new_price}")
