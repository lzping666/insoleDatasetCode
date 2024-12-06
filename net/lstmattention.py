import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import r2_score
import torch.nn.functional as F
from caculate.fsr_processor import load_fsr_data
from caculate.pose_processor import load_pose_data
from torch.optim.lr_scheduler import ReduceLROnPlateau


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


class LSTMAttentionModel(nn.Module):
    def __init__(self, seq_input_size, static_input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(LSTMAttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(seq_input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)

        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # 合并静态特征的全连接层
        self.fc_combine = nn.Linear(hidden_size + static_input_size, hidden_size)

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x_seq, x_static):
        # LSTM处理序列数据
        lstm_out, _ = self.lstm(x_seq)  # shape: (batch, seq_len, hidden_size)

        # 注意力机制
        attention_weights = self.attention(lstm_out)  # shape: (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # shape: (batch, hidden_size)

        # 合并静态特征
        combined = torch.cat((context_vector, x_static), dim=1)
        combined = self.fc_combine(combined)
        combined = F.relu(combined)

        # 输出层 - 扩展为序列长度
        output = self.fc(combined)  # shape: (batch, output_size)
        output = output.unsqueeze(1).repeat(1, x_seq.size(1), 1)  # shape: (batch, seq_len, output_size)

        return output

class PyTorchFSRPredictor:
    def __init__(self, seq_input_size, static_input_size, hidden_size, num_layers,
                 output_size, learning_rate=0.001, batch_size=32, num_epochs=100, dropout_rate=0.2):
        self.seq_input_size = seq_input_size
        self.static_input_size = static_input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate
        self.device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu")
        self.model = self.build_model()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)

    def build_model(self):
        return LSTMAttentionModel(
            seq_input_size=self.seq_input_size,
            static_input_size=self.static_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size,
            dropout_rate=self.dropout_rate
        ).to(self.device)



    # 准备用于训练和验证的序列数据 姿态数据从三维（frames, 33, 3）展平为二维（frames, 99）
    def prepare_sequence_data(self, pose_data, fsr_data, sequence_length, static_features, test_size=0.2):
        X_sequences = []
        y_sequences = []

        # 展平姿态数据
        pose_features = pose_data.reshape(pose_data.shape[0], -1)

        # 使用MinMaxScaler来确保数据在0-1之间
        from sklearn.preprocessing import MinMaxScaler
        scaler_pose = MinMaxScaler(feature_range=(0, 1))
        scaler_fsr = MinMaxScaler(feature_range=(0, 1))

        pose_features_scaled = scaler_pose.fit_transform(pose_features)
        fsr_scaled = scaler_fsr.fit_transform(fsr_data)

        # 保存scaler为类属性，用于后续反转换
        self.scaler_fsr = scaler_fsr
        self.scaler_pose = scaler_pose

        # 创建序列
        stride = 2
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
        X_train_seq = torch.FloatTensor(X_sequences[train_indices]).to(self.device)
        y_train = torch.FloatTensor(y_sequences[train_indices]).to(self.device)
        X_test_seq = torch.FloatTensor(X_sequences[test_indices]).to(self.device)
        y_test = torch.FloatTensor(y_sequences[test_indices]).to(self.device)

        # 数据增强
        X_train_seq_augmented = self.add_noise(X_train_seq)
        y_train_augmented = y_train

        # 合并原始数据和增强数据
        X_train_seq = torch.cat([X_train_seq, X_train_seq_augmented], dim=0)
        y_train = torch.cat([y_train, y_train_augmented], dim=0)

        # 准备静态特征
        static_features = static_features.to(self.device)
        # 扩展静态特征以匹配序列数据的批次大小
        X_train_static = static_features.unsqueeze(0).repeat(len(X_train_seq), 1)
        X_test_static = static_features.unsqueeze(0).repeat(len(X_test_seq), 1)

        # 创建包含序列特征和静态特征的数据集
        train_dataset = TensorDataset(X_train_seq, X_train_static, y_train)
        test_dataset = TensorDataset(X_test_seq, X_test_static, y_test)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader, scaler_pose, scaler_fsr

    # 修改训练过程以处理注意力权重
    def train_model(self, train_loader, val_loader):
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(self.num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            for batch_seq, batch_static, batch_y in train_loader:
                self.optimizer.zero_grad()

                outputs = self.model(batch_seq, batch_static)
                # 确保输出和目标具有相同的维度
                batch_y = batch_y.view(outputs.shape)

                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            # 验证阶段
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_seq, batch_static, batch_y in val_loader:
                    outputs = self.model(batch_seq, batch_static)
                    batch_y = batch_y.view(outputs.shape)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()

            # 更新学习率
            self.scheduler.step(val_loss)

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            if epoch % 10 == 0:
                print(f'Epoch {epoch}/{self.num_epochs}, Training Loss: {train_loss / len(train_loader):.4f}, Validation Loss: {val_loss / len(val_loader):.4f}')



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



    def process_sequence(self, pose_data, fsr_data, static_features):
        try:
            sequence_length = 32
            pose_features = pose_data.reshape(pose_data.shape[0], -1)

            X_sequences = []
            for i in range(len(pose_features) - sequence_length + 1):
                X_sequences.append(pose_features[i:i + sequence_length])

            X_sequences = np.array(X_sequences)

            # 使用保存的scaler进行转换
            X_sequences_reshaped = X_sequences.reshape(-1, pose_data.shape[1] * pose_data.shape[2])
            X_scaled = self.scaler_pose.transform(X_sequences_reshaped)
            X_scaled = X_scaled.reshape(X_sequences.shape)

            # 将序列数据转换为张量
            X_seq = torch.FloatTensor(X_scaled).to(self.device)

            # 准备静态特征
            static_features = torch.FloatTensor(static_features).to(self.device)
            # 扩展静态特征以匹配序列数据的批次大小
            X_static = static_features.unsqueeze(0).repeat(len(X_seq), 1)

            self.model = self.model.to(self.device)
            self.model.eval()

            batch_size = 32  # 根据GPU内存调整
            predictions = []

            for i in range(0, len(X_seq), batch_size):
                batch_seq = X_seq[i:i + batch_size]
                batch_static = X_static[i:i + batch_size]
                with torch.no_grad():
                    # 模型只返回一个输出
                    batch_predictions = self.model(batch_seq, batch_static)
                    predictions.append(batch_predictions.cpu().numpy())

            predicted_sequences = np.concatenate(predictions, axis=0)

            # 检查预测结果的维度
            if len(predicted_sequences.shape) == 3:
                predicted_fsr = predicted_sequences[:, -1, :]  # 如果是3D张量
            else:
                predicted_fsr = predicted_sequences  # 如果是2D张量

            # 添加padding
            padding = np.zeros((sequence_length - 1, predicted_fsr.shape[1]))
            predicted_fsr = np.vstack([padding, predicted_fsr])

            # 反归一化
            predicted_fsr_reshaped = predicted_fsr.reshape(-1, fsr_data.shape[1])
            predicted_fsr = self.scaler_fsr.inverse_transform(predicted_fsr_reshaped)
            # 剪裁负值，确保数据合理
            predicted_fsr = np.clip(predicted_fsr, self.scaler_fsr.data_min_, self.scaler_fsr.data_max_)
            predicted_fsr_smoothed = self.moving_average(predicted_fsr, window_size=10)

            return predicted_fsr_smoothed  # 只返回预测值

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
            mape = np.mean(np.abs((true_fsr[mask] - predicted_fsr[mask]) / true_fsr[mask])) * 100
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
    # 加载数据
    csv_file_path = "../datasets/filtered_insole_data4.csv"
    timestamps, fsr_values = load_fsr_data(csv_file_path)
    processed_fsr = preprocess_fsr_data(fsr_values)

    pose_csv_file_path = "../datasets/pose3.csv"
    pose_data = load_pose_data(pose_csv_file_path)
    processed_pose_data = preprocess_pose_data(pose_data)

    # 确保数据长度一致
    num_samples = min(processed_pose_data.shape[0], processed_fsr.shape[0])
    processed_pose_data = processed_pose_data[:num_samples]
    processed_fsr = processed_fsr[:num_samples]

    # 设置序列长度
    sequence_length = 32

    # 原始静态特征
    weight = 100  # 体重(kg)
    height = 178  # 身高(cm)
    shoe_size = 40  # 鞋码

    # 手动归一化静态特征
    weight_range = (40, 120)
    height_range = (150, 190)
    shoe_size_range = (35, 45)

    normalized_weight = (weight - weight_range[0]) / (weight_range[1] - weight_range[0])
    normalized_height = (height - height_range[0]) / (height_range[1] - height_range[0])
    normalized_shoe_size = (shoe_size - shoe_size_range[0]) / (shoe_size_range[1] - shoe_size_range[0])

    # 组合归一化后的静态特征
    static_features = torch.tensor([normalized_weight, normalized_height, normalized_shoe_size], dtype=torch.float32)

    # 初始化预测器
    predictor = PyTorchFSRPredictor(
        seq_input_size=processed_pose_data.shape[1] * processed_pose_data.shape[2],  # 序列特征维度
        static_input_size=len(static_features),  # 静态特征维度
        hidden_size=128,
        num_layers=4,
        output_size=processed_fsr.shape[1],  # FSR传感器数量
        learning_rate=0.001,
        batch_size=64,
        num_epochs=100,
        dropout_rate=0.2
    )

    # 准备序列数据
    train_loader, val_loader, scaler_pose, scaler_fsr = predictor.prepare_sequence_data(
        processed_pose_data, processed_fsr, sequence_length, static_features)

    # 训练模型
    predictor.train_model(train_loader, val_loader)

    # 预测和评估
    predicted_fsr = predictor.process_sequence(processed_pose_data, processed_fsr, static_features)

    # 确保长度一致
    min_length = min(len(processed_fsr), len(predicted_fsr))
    processed_fsr = processed_fsr[:min_length]
    predicted_fsr = predicted_fsr[:min_length]

    # 计算评估指标
    mse, rmse, mae, r2, mape = predictor.evaluate_model(processed_fsr, predicted_fsr)

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAPE: {mape:.4f}" + '%')

    # 绘制结果
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    for i in range(4):
        row = i // 2
        col = i % 2
        axs[row, col].plot(processed_fsr[:, i], label='True FSR')
        axs[row, col].plot(predicted_fsr[:, i], label='Predicted FSR')
        axs[row, col].set_title(f'FSR Prediction Results for Sensor {i + 1}')
        axs[row, col].set_xlabel('Time steps')
        axs[row, col].set_ylabel('FSR values')
        axs[row, col].legend()

    plt.tight_layout()
    plt.show()