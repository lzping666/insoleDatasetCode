import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import r2_score

from caculate.fsr_processor import load_fsr_data
from caculate.pose_processor import load_pose_data


def preprocess_fsr_data(fsr_values, smooth_window=5):
    """
    预处理FSR数据

    Parameters:
    fsr_values: 原始FSR数据
    smooth_window: 平滑窗口大小

    Returns:
    processed_values: 处理后的FSR数据
    """
    # 移动平均平滑
    processed_values = pd.DataFrame(fsr_values).rolling(
        window=smooth_window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values


    return processed_values


def preprocess_pose_data(pose_data, smooth_window=5):
    """
    预处理姿态数据
    Parameters:
    pose_data: 原始姿态数据，shape: (frames, 33, 3)
    smooth_window: 平滑窗口大小

    Returns:
    processed_pose: 处理后的姿态数据，shape: (frames, 33, 3)
    """
    frames, num_keypoints, num_coordinates = pose_data.shape

    # 直接使用原始数据，不需要reshape
    pose_values = pose_data

    # 对每个维度应用移动平均平滑
    processed_pose = np.zeros_like(pose_values)
    for i in range(num_keypoints):
        for j in range(num_coordinates):
            values = pd.Series(pose_values[:, i, j])
            smoothed = values.rolling(window=smooth_window, center=True).mean()
            processed_pose[:, i, j] = smoothed.fillna(method='bfill').fillna(method='ffill')

    return processed_pose
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_norm1 = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Apply batch normalization
        x = x.transpose(1, 2)  # (batch_size, input_size, seq_len)
        x = self.batch_norm1(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, input_size)

        # LSTM layers
        lstm_out, _ = self.lstm(x)

        # Apply dropout and batch normalization to the output
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.transpose(1, 2)  # (batch_size, hidden_size, seq_len)
        lstm_out = self.batch_norm2(lstm_out)
        lstm_out = lstm_out.transpose(1, 2)  # (batch_size, seq_len, hidden_size)

        # Output layer
        output = self.fc(lstm_out)
        return output

class PyTorchFSRPredictor:
    def __init__(self, hidden_size, num_layers, learning_rate=0.001, batch_size=32, num_epochs=200, dropout_rate=0.2):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate
        self.model = None
        self.device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu")
    def build_model(self, input_size, output_size):
        return LSTMModel(input_size, self.hidden_size, self.num_layers, output_size, self.dropout_rate)



    # 准备用于训练和验证的序列数据 姿态数据从三维（frames, 33, 3）展平为二维（frames, 99）
    def prepare_sequence_data(self, pose_data, fsr_data, sequence_length, test_size=0.2):
        X_sequences = []
        y_sequences = []

        # 展平姿态数据
        pose_features = pose_data.reshape(pose_data.shape[0], -1)

        # 使用MinMaxScaler来确保数据在0-1之间
        from sklearn.preprocessing import MinMaxScaler
        scaler_pose = MinMaxScaler(feature_range=(0, 1))  # 将StandardScaler改为MinMaxScaler
        scaler_fsr = MinMaxScaler(feature_range=(0, 1))

        pose_features_scaled = scaler_pose.fit_transform(pose_features)
        fsr_scaled = scaler_fsr.fit_transform(fsr_data)

        # 保存scaler为类属性，用于后续反转换
        self.scaler_fsr = scaler_fsr
        self.scaler_pose = scaler_pose  # 保存pose_scaler

        # 创建序列
        stride = 1
        for i in range(0, len(pose_features_scaled) - sequence_length + 1, stride):
            X_sequences.append(pose_features_scaled[i:i + sequence_length])
            y_sequences.append(fsr_scaled[i:i + sequence_length])

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)


        # Train-test split with shuffle
        indices = np.arange(len(X_sequences))
        np.random.shuffle(indices)
        train_size = int(len(indices) * (1 - test_size))

        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        # 创建数据加载器
        X_train = torch.FloatTensor(X_sequences[train_indices]).to(self.device)
        y_train = torch.FloatTensor(y_sequences[train_indices]).to(self.device)
        X_test = torch.FloatTensor(X_sequences[test_indices]).to(self.device)
        y_test = torch.FloatTensor(y_sequences[test_indices]).to(self.device)

        # 数据增强
        X_train_augmented = self.add_noise(X_train)
        y_train_augmented = y_train

        # 合并原始数据和增强数据
        X_train = torch.cat([X_train, X_train_augmented], dim=0)
        y_train = torch.cat([y_train, y_train_augmented], dim=0)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader, scaler_pose, scaler_fsr



    def train_model(self, train_loader, val_loader):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                      verbose=True)  # Adjust learning rate
        patience = 10  # Early stopping patience
        best_val_loss = float('inf')
        early_stop_counter = 0

        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    val_loss += criterion(outputs, batch_y).item()

            # Compute average losses
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            # Scheduler step
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0  # Reset counter if validation loss improves
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"Early stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                    break

            # Print progress
            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        print("Training complete.")

    def moving_average(self, data, window_size=10):
        """
        计算移动平均
        """
        weights = np.ones(window_size) / window_size
        # 对每个传感器单独进行移动平均
        smoothed_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            smoothed_data[:, i] = np.convolve(data[:, i], weights, mode='same')
        return smoothed_data

    def add_noise(self, sequence, noise_factor=0.01):
        noise = torch.randn_like(sequence) * noise_factor
        return sequence + noise


    def process_sequence(self, pose_data, fsr_data):
        try:
            sequence_length = 32
            pose_features = pose_data.reshape(pose_data.shape[0], -1)

            X_sequences = []
            for i in range(len(pose_features) - sequence_length + 1):
                X_sequences.append(pose_features[i:i + sequence_length])

            X_sequences = np.array(X_sequences)

            # Use the saved scaler for transformation
            X_sequences_reshaped = X_sequences.reshape(-1, pose_data.shape[1] * pose_data.shape[2])
            X_scaled = self.scaler_pose.transform(X_sequences_reshaped)
            X_scaled = X_scaled.reshape(X_sequences.shape)

            X = torch.FloatTensor(X_scaled).to(self.device)

            self.model = self.model.to(self.device)
            self.model.eval()

            batch_size = 32  # Adjust based on your GPU memory
            predictions = []

            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size]
                with torch.no_grad():
                    batch_predictions = self.model(batch)
                    batch_predictions = batch_predictions.cpu().numpy()
                    predictions.append(batch_predictions)

            predicted_sequences = np.concatenate(predictions, axis=0)

            predicted_fsr = predicted_sequences[:, -1, :]

            padding = np.zeros((sequence_length - 1, predicted_fsr.shape[1]))
            predicted_fsr = np.vstack([padding, predicted_fsr])

            # 反归一化
            predicted_fsr_reshaped = predicted_fsr.reshape(-1, fsr_data.shape[1])
            predicted_fsr = self.scaler_fsr.inverse_transform(predicted_fsr_reshaped)
            # **剪裁负值，确保数据合理**
            predicted_fsr = np.clip(predicted_fsr, self.scaler_fsr.data_min_, self.scaler_fsr.data_max_)
            return predicted_fsr

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise




    def evaluate_model(self, true_fsr, predicted_fsr):
        """评估模型性能"""
        # 确保数据形状一致
        assert true_fsr.shape == predicted_fsr.shape, "Shape mismatch between true and predicted values"

        # 计算评估指标（直接基于原始数据）
        mse = np.mean((true_fsr - predicted_fsr) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(true_fsr - predicted_fsr))

        # 计算相对误差 (MAPE: Mean Absolute Percentage Error)
        # 只在true_fsr大于阈值时计算相对误差
        # 设置范围
        lower_threshold = 0
        upper_threshold = 1000
        mask = true_fsr > (true_fsr > lower_threshold) & (true_fsr < upper_threshold)
        if np.any(mask):
            mape = np.mean(np.abs((true_fsr[mask] - predicted_fsr[mask]) / true_fsr[mask]) ) * 100
        else:
            mape = float('nan')


        # 计算 R² 分数
        r2_scores = []
        for i in range(true_fsr.shape[1]):
            r2 = r2_score(true_fsr[:, i], predicted_fsr[:, i])
            r2_scores.append(r2)
        r2 = np.mean(r2_scores)

        return mse, rmse, mae, r2,mape



def moving_average(data, window_size, axis=0):
    if window_size <= 0:
        raise ValueError("Window size must be positive.")
    if window_size == 1:
        return data

    smoothed_data = np.apply_along_axis(lambda m: np.convolve(m, np.ones(window_size), 'same') / window_size, axis, data)
    return smoothed_data

if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    csv_file_path = "../datasets/filtered_insole_data4.csv"
    timestamps, fsr_values = load_fsr_data(csv_file_path)
    processed_fsr = preprocess_fsr_data(fsr_values)

    pose_csv_file_path = "../datasets/pose3.csv"
    pose_data = load_pose_data(pose_csv_file_path)
    processed_pose_data = preprocess_pose_data(pose_data)
    # print(f"Processed pose data range: min={processed_pose_data.min()}, max={processed_pose_data.max()}")
    # print(f"Processed fsr data range: min={processed_fsr.min()}, max={processed_fsr.max()}")
    num_samples = min(processed_pose_data.shape[0], processed_fsr.shape[0])
    processed_pose_data = processed_pose_data[:num_samples]
    processed_fsr = processed_fsr[:num_samples]

    # 设置序列长度
    sequence_length = 32

    predictor = PyTorchFSRPredictor(
        hidden_size=256,  # 增加隐藏层大小
        num_layers=5,  # 增加层数
        learning_rate=0.001,  # 减小学习率
        batch_size=128,  # 增加批次大小
        num_epochs=200,
        dropout_rate=0.3  # 添加dropout
    )


    # 准备序列数据
    train_loader, val_loader, scaler_pose, scaler_fsr = predictor.prepare_sequence_data(
        processed_pose_data, processed_fsr, sequence_length)

    # 构建和训练模型
    input_size = processed_pose_data.shape[1] * processed_pose_data.shape[2]  # 33*3=99 特征
    output_size = processed_fsr.shape[1]  # FSR传感器数量
    predictor.model = predictor.build_model(input_size, output_size).to(predictor.device)
    predictor.train_model(train_loader, val_loader)

    # 预测和评估
    predicted_fsr = predictor.process_sequence(processed_pose_data, processed_fsr)

    # 确保长度一致
    min_length = min(len(processed_fsr), len(predicted_fsr))
    processed_fsr = processed_fsr[:min_length]
    predicted_fsr = predicted_fsr[:min_length]

    mse, rmse, mae, r2, mape = predictor.evaluate_model(processed_fsr, predicted_fsr)

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAPE: {mape:.4f}" + '%')


    # Plotting the results
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    for i in range(4):
        row = i // 2
        col = i % 2
        axs[row, col].plot(processed_fsr[:, i], label='True FSR')
        axs[row, col].plot(predicted_fsr[:, i], label='Predicted FSR')
        axs[row, col].set_title(f'FSR Prediction Results for Sensor {i+1}')
        axs[row, col].set_xlabel('Time steps')
        axs[row, col].set_ylabel('FSR values')
        axs[row, col].legend()

    plt.tight_layout()
    plt.show()