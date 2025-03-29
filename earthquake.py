import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 1. 自动获取 USGS 地震数据
url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
params = {
    "format": "geojson",
    "starttime": "1900-01-01",
    "endtime": "2024-03-01",
    "minmagnitude": 7,
    "limit": 10000  # 获取最多 10000 条数据
}

try:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()  # 检查 HTTP 请求是否成功
    data = response.json()
except requests.exceptions.RequestException as e:
    print(f"数据获取失败: {e}")
    exit()

# 2. 解析数据
earthquakes = []
for eq in data['features']:
    time = pd.to_datetime(eq['properties']['time'], unit='ms')
    earthquakes.append(time)

df = pd.DataFrame(earthquakes, columns=["Date"])
df = df.sort_values(by="Date").reset_index(drop=True)

# 3. 计算地震发生的时间间隔（单位：天）
df["Interval"] = df["Date"].diff().dt.days
df = df.dropna()

# 4. 训练 ARIMA 模型
model = ARIMA(df["Interval"], order=(1,1,1))  # 选择 ARIMA(1,1,1) 作为示例
model_fit = model.fit()

# 5. 预测未来下一个地震的间隔天数
forecast_days = model_fit.forecast(steps=1).iloc[0]
last_date = df["Date"].iloc[-1]
predicted_date = last_date + pd.Timedelta(days=int(forecast_days))

print(f"预计下次7级以上地震发生的时间: {predicted_date.date()}")

# 6. 可视化历史地震间隔趋势
plt.figure(figsize=(12, 5))
plt.plot(df["Date"][1:], df["Interval"], marker="o", label="历史地震间隔")
plt.axvline(x=predicted_date, color='r', linestyle='--', label=f"预测地震({predicted_date.date()})")
plt.xlabel("年份")
plt.ylabel("地震间隔（天）")
plt.title("历史地震间隔趋势与预测")
plt.legend()
plt.xticks(rotation=45)
plt.show()
