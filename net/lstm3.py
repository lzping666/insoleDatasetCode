import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import r2_score

from caculate.fsr_processor import load_fsr_data, preprocess_fsr_data
from caculate.pose_processor import load_pose_data,preprocess_pose_data


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

    def prepare_sequence_data(self, pose_data, fsr_data, sequence_length, test_size=0.2):
        X_sequences = []
        y_sequences = []

        # 展平姿态数据
        pose_features = pose_data.reshape(pose_data.shape[0], -1)

        # 创建序列
        stride = 2  # You can adjust the stride
        for i in range(0, len(pose_features) - sequence_length + 1, stride):
            X_sequences.append(pose_features[i:i + sequence_length])
            y_sequences.append(fsr_data[i:i + sequence_length])

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)

        # Reshape for scaling (important for correct scaling)
        X_sequences_reshaped = X_sequences.reshape(-1, pose_data.shape[1] * pose_data.shape[2])
        y_sequences_reshaped = y_sequences.reshape(-1, fsr_data.shape[1])

        self.scaler_pose = MinMaxScaler(feature_range=(0, 1))
        self.scaler_fsr = MinMaxScaler(feature_range=(0, 1))

        X_sequences_scaled = self.scaler_pose.fit_transform(X_sequences_reshaped)
        y_sequences_scaled = self.scaler_fsr.fit_transform(y_sequences_reshaped)

        # Reshape back to sequences
        X_sequences_scaled = X_sequences_scaled.reshape(X_sequences.shape)
        y_sequences_scaled = y_sequences_scaled.reshape(y_sequences.shape)


        # Train-test split with shuffle
        indices = np.arange(len(X_sequences_scaled))
        np.random.shuffle(indices)
        train_size = int(len(indices) * (1 - test_size))

        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        # Create data loaders
        X_train = torch.FloatTensor(X_sequences_scaled[train_indices]).to(self.device)
        y_train = torch.FloatTensor(y_sequences_scaled[train_indices]).to(self.device)
        X_test = torch.FloatTensor(X_sequences_scaled[test_indices]).to(self.device)
        y_test = torch.FloatTensor(y_sequences_scaled[test_indices]).to(self.device)

        # Data augmentation (if needed)
        X_train_augmented = self.add_noise(X_train)
        y_train_augmented = y_train

        # Combine original and augmented data
        X_train = torch.cat([X_train, X_train_augmented], dim=0)
        y_train = torch.cat([y_train, y_train_augmented], dim=0)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader, self.scaler_pose, self.scaler_fsr


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
            sequence_length = 48
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
            predicted_fsr_smoothed = self.moving_average(predicted_fsr, window_size=10)
            return predicted_fsr_smoothed

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise

    def evaluate_model(self, true_fsr, predicted_fsr):
        """
        评估模型性能
        Parameters:
        true_fsr: 真实的FSR数据
        predicted_fsr: 模型预测的FSR数据

        Returns:
        mse: 均方误差
        rmse: 均方根误差
        mae: 平均绝对误差
        r2: R平方分数
        """
        # 确保数据形状一致
        assert true_fsr.shape == predicted_fsr.shape, "Shape mismatch between true and predicted values"
        # 计算评估指标
        mse = np.mean((true_fsr - predicted_fsr) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(true_fsr - predicted_fsr))

        # 计算R2分数
        r2_scores = []
        for i in range(true_fsr.shape[1]):
            r2 = r2_score(true_fsr[:, i], predicted_fsr[:, i])
            r2_scores.append(r2)
        r2 = np.mean(r2_scores)

        return mse, rmse, mae, r2

def moving_average(data, window_size, axis=0):
    if window_size <= 0:
        raise ValueError("Window size must be positive.")
    if window_size == 1:
        return data

    smoothed_data = np.apply_along_axis(lambda m: np.convolve(m, np.ones(window_size), 'same') / window_size, axis, data)
    return smoothed_data

def inverse_transform(scaler, data):
    """
    使用Scaler对数据进行反归一化

    Parameters:
    scaler: 预先保存的MinMaxScaler对象
    data: 需要反归一化的数据 (归一化后的)

    Returns:
    反归一化后的数据
    """
    return scaler.inverse_transform(data)

if __name__ == '__main__':

    # Load FSR data
    csv_file_path = "../datasets/downsampled_insole_data4.csv"
    timestamps, fsr_values = load_fsr_data(csv_file_path)

    # Load pose data
    pose_csv_file_path = "../datasets/pose3.csv"
    pose_data = load_pose_data(pose_csv_file_path)

    # Ensure pose_data and fsr_data have the same number of samples
    num_samples = min(pose_data.shape[0], fsr_values.shape[0])
    pose_data = pose_data[:num_samples]
    fsr_values = fsr_values[:num_samples]
    timestamps = timestamps[:num_samples]  # Synchronize and trim the length of timestamps

    # Initialize the predictor
    predictor = PyTorchFSRPredictor(
        hidden_size=256,  # Hidden layer size
        num_layers=4,  # Number of LSTM layers
        learning_rate=0.001,  # Learning rate
        batch_size=128,  # Batch size
        num_epochs=200,  # Number of epochs
        dropout_rate=0.3  # Dropout rate
    )

    # Preprocess FSR data (smoothing)
    smoothed_fsr_values = predictor.moving_average(fsr_values, window_size=10)

    # Prepare sequence data
    sequence_length = 48
    train_loader, val_loader, scaler_pose, scaler_fsr = predictor.prepare_sequence_data(
        pose_data, smoothed_fsr_values, sequence_length
    )

    # Print debug information: verify normalization ranges

    # Build and train the model
    input_size = pose_data.shape[1] * pose_data.shape[2]  # 33*3=99 features
    output_size = fsr_values.shape[1]  # Number of FSR sensors
    predictor.model = predictor.build_model(input_size, output_size).to(predictor.device)
    predictor.train_model(train_loader, val_loader)

    # Model inference
    predicted_fsr = predictor.process_sequence(pose_data, smoothed_fsr_values)

    # Evaluate the model
    mse, rmse, mae, r2 = predictor.evaluate_model(fsr_values, predicted_fsr)

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # Plotting the results
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    for i in range(4):
        row = i // 2
        col = i % 2
        axs[row, col].plot(timestamps, fsr_values[:, i], label='Original FSR Data')
        axs[row, col].plot(timestamps, predicted_fsr[:, i], label='Predicted FSR Data')
        axs[row, col].set_title(f'FSR Prediction Results for Sensor {i+1}')
        axs[row, col].set_xlabel('Timestamps')
        axs[row, col].set_ylabel('FSR Value')
        axs[row, col].legend()

    plt.tight_layout()
    plt.show()