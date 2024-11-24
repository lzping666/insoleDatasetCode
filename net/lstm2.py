import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from caculate.fsr_processor import load_fsr_data, preprocess_fsr_data
from caculate.pose_processor import load_pose_data


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

    # 对每个关键点的坐标进行归一化
    for i in range(num_keypoints):
        for j in range(num_coordinates):
            min_val = processed_pose[:, i, j].min()
            max_val = processed_pose[:, i, j].max()
            if max_val > min_val:  # 避免除以零
                processed_pose[:, i, j] = (processed_pose[:, i, j] - min_val) / (max_val - min_val)

    return processed_pose




class PyTorchFSRPredictor:
    def __init__(self, hidden_size, num_layers, learning_rate=0.001, batch_size=32, num_epochs=200):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = None
        self.device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu")

    def build_model(self, input_size, output_size):
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                out = self.fc(lstm_out[:, -1, :])  # Use the last time step's output
                return out

        return LSTMModel(input_size, self.hidden_size, self.num_layers, output_size)


    def prepare_data(self, pose_data, fsr_data, test_size=0.2):
        # Flatten pose data for standardization
        pose_data_2d = pose_data.reshape(pose_data.shape[0], -1)

        # Train-test split
        n_samples = pose_data.shape[0]
        n_test = int(n_samples * test_size)
        n_train = n_samples - n_test

        train_indices = range(n_train)
        test_indices = range(n_train, n_samples)

        # Standardize data
        scaler_pose = StandardScaler()
        pose_train_scaled = scaler_pose.fit_transform(pose_data_2d[train_indices])
        pose_test_scaled = scaler_pose.transform(pose_data_2d[test_indices])

        scaler_fsr = StandardScaler()
        fsr_train_scaled = scaler_fsr.fit_transform(fsr_data[train_indices])
        fsr_test_scaled = scaler_fsr.transform(fsr_data[test_indices])

        # Reshape pose data
        pose_train_scaled = pose_train_scaled.reshape(n_train, *pose_data.shape[1:])
        pose_test_scaled = pose_test_scaled.reshape(n_test, *pose_data.shape[1:])

        # Convert to tensors
        X_train = torch.FloatTensor(pose_train_scaled).to(self.device)
        y_train = torch.FloatTensor(fsr_train_scaled).to(self.device)
        X_test = torch.FloatTensor(pose_test_scaled).to(self.device)
        y_test = torch.FloatTensor(fsr_test_scaled).to(self.device)

        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader, scaler_pose, scaler_fsr

    def train_model(self, train_loader, val_loader):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

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

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    def process_sequence(self, pose_data, fsr_data):
        train_loader, val_loader, scaler_pose, scaler_fsr = self.prepare_data(pose_data, fsr_data)
        input_size = pose_data.shape[2]  # Feature size
        output_size = fsr_data.shape[1]
        self.model = self.build_model(input_size, output_size).to(self.device)
        self.train_model(train_loader, val_loader)

        self.model.eval()
        with torch.no_grad():
            pose_data_scaled = scaler_pose.transform(pose_data.reshape(pose_data.shape[0], -1)).reshape(pose_data.shape)
            X = torch.FloatTensor(pose_data_scaled).to(self.device)

            predicted_fsr_scaled = self.model(X)
            predicted_fsr = scaler_fsr.inverse_transform(predicted_fsr_scaled.cpu().numpy())
            predicted_fsr_smoothed = moving_average(predicted_fsr, window_size=10, axis=0)

            return predicted_fsr_smoothed

    def evaluate_model(self, true_fsr, predicted_fsr):
        mse = np.mean((true_fsr - predicted_fsr) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(true_fsr - predicted_fsr))
        r2 = r2_score(true_fsr, predicted_fsr)

        print(f"模型评估结果:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")

        return mse, rmse, mae, r2

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

    csv_file_path = "../datasets/downsampled_insole_data4.csv"
    timestamps, fsr_values = load_fsr_data(csv_file_path)
    processed_fsr = preprocess_fsr_data(fsr_values)

    pose_csv_file_path = "../datasets/pose2.csv"
    pose_data = load_pose_data(pose_csv_file_path)
    processed_pose_data = preprocess_pose_data(pose_data)

    num_samples = min(processed_pose_data.shape[0], processed_fsr.shape[0])
    processed_pose_data = processed_pose_data[:num_samples]
    processed_fsr = processed_fsr[:num_samples]

    predictor = PyTorchFSRPredictor(
        hidden_size=64,
        num_layers=2,
        learning_rate=0.001,
        batch_size=128,
        num_epochs=100
    )

    predicted_fsr = predictor.process_sequence(processed_pose_data, processed_fsr)
    mse, rmse, mae, r2 = predictor.evaluate_model(processed_fsr, predicted_fsr)


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