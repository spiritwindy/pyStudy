import sqlite3
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

class EarthquakeDataset(Dataset):
    def __init__(self, db_path, seq_length=30, pred_length=5):
        """
        地震数据集加载器
        
        参数:
            db_path: SQLite数据库路径
            seq_length: 输入序列长度(天数)
            pred_length: 预测序列长度(天数)
        """
        self.seq_length = seq_length
        self.pred_length = pred_length
        
        # 从数据库加载数据
        conn = sqlite3.connect(db_path)
        query = "SELECT magnitude, time, latitude, longitude FROM Earthquakes ORDER BY time"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # 转换时间格式并排序
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        
        # 特征工程
        df['day_of_year'] = df['time'].dt.dayofyear
        df['days_since_last'] = df['time'].diff().dt.days.fillna(0)
        
        # 归一化处理
        self.scalers = {}
        for col in ['magnitude', 'latitude', 'longitude', 'day_of_year', 'days_since_last']:
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]])
            self.scalers[col] = scaler
        
        # 转换为numpy数组
        self.data = df[['magnitude', 'latitude', 'longitude', 'day_of_year', 'days_since_last']].values
        self.timestamps = df['time'].values
        
    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length + 1
    
    def __getitem__(self, idx):
        # 获取输入序列和预测序列
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+self.seq_length:idx+self.seq_length+self.pred_length, 0]  # 只预测震级
        
        # 转换为torch张量
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        
        return x, y
    
if __name__ == '__main__':
    # 测试数据集加载器
    db_path = 'e:\pyStudy\database.sqlite'
    dataset = EarthquakeDataset(db_path, seq_length=30, pred_length=5)
    print(len(dataset))
    x, y = dataset[0]
    print(x.shape, y.shape)
