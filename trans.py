def train_model(db_path, epochs=50, batch_size=32, seq_length=30, pred_length=5):
    # 准备数据集
    dataset = EarthquakeDataset(db_path, seq_length, pred_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatioTemporalTransformer(
        input_dim=5,
        pred_length=pred_length
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    return model, dataset.scalers

def predict_future_earthquakes(model, scalers, last_sequence, pred_days=5):
    """
    预测未来地震
    
    参数:
        model: 训练好的模型
        scalers: 用于反归一化的scaler字典
        last_sequence: 最后N天的地震数据 (numpy数组)
        pred_days: 要预测的天数
    """
    device = next(model.parameters()).device
    
    # 预处理输入序列
    seq_length = len(last_sequence)
    model.eval()
    
    with torch.no_grad():
        # 转换为tensor并添加batch维度
        input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
        
        # 预测
        predictions = model(input_tensor).cpu().numpy()[0]
    
    # 反归一化预测结果
    magnitude_scaler = scalers['magnitude']
    predictions = magnitude_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    return predictions

# 示例使用
if __name__ == "__main__":
    # 训练模型
    model, scalers = train_model("earthquakes.db", epochs=30)
    
    # 获取最后一段序列用于预测
    dataset = EarthquakeDataset("earthquakes.db")
    last_sequence = dataset.data[-dataset.seq_length:]
    
    # 预测未来5天的地震
    predictions = predict_future_earthquakes(model, scalers, last_sequence)
    print("未来5天的预测震级:", predictions)