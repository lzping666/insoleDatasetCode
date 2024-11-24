import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, Subset
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch.nn.functional as F

from caculate.fsr_processor import load_fsr_data, preprocess_fsr_data




def load_and_preprocess_pose_data(pose_csv_path):
    """加载和预处理姿态数据"""
    # 读取CSV文件
    df = pd.read_csv(pose_csv_path)

    # 提取x, y, z坐标列
    coords_columns = []
    for i in range(33):  # 33个关键点
        coords_columns.extend([f'video2_x{i}', f'video2_y{i}', f'video2_z{i}'])

    # 提取坐标数据
    pose_data = df[coords_columns].values

    # 重塑数据为 (samples, 3, 33) 的形状
    # 3 表示 x,y,z 坐标，33 表示关键点数量
    samples = pose_data.shape[0]
    pose_data = pose_data.reshape(samples, 33, 3)
    pose_data = pose_data.transpose(0, 2, 1)  # 转换为 (samples, 3, 33)

    return pose_data


class PyTorchFSRPredictor:
    def __init__(self, hidden_size=128, learning_rate=0.01, batch_size=32, num_epochs=100):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = None
        self.device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu")
        print(f"Using device: {self.device}")

    def build_model(self, input_size, output_size):
        print(f"Building model with input_size: {input_size}, output_size: {output_size}")

        class BasicRNN(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, dropout=0.3):
                super(BasicRNN, self).__init__()

                self.actual_input_size = 3
                print(f"Actual input size: {self.actual_input_size}")

                self.rnn = nn.RNN(
                    input_size=self.actual_input_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    nonlinearity='relu'
                )

                self.bn1 = nn.BatchNorm1d(hidden_size)
                self.dropout = nn.Dropout(dropout)
                self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
                self.bn2 = nn.BatchNorm1d(hidden_size // 2)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size // 2, output_size)

                # 修改后的权重初始化
                self.init_weights()

            def init_weights(self):
                for name, param in self.named_parameters():
                    if 'weight_ih' in name:  # RNN input-hidden weights
                        nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
                    elif 'weight_hh' in name:  # RNN hidden-hidden weights
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:  # 所有偏置项
                        nn.init.constant_(param, 0.0)
                    elif len(param.shape) >= 2:  # 线性层权重
                        nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')

            def forward(self, x):
                x = x.transpose(1, 2)

                # RNN层
                rnn_out, _ = self.rnn(x)
                out = rnn_out[:, -1, :]

                # 添加残差连接
                identity = out

                # 第一个全连接层
                out = self.bn1(out)
                out = self.dropout(out)
                out = self.fc1(out)
                out = self.bn2(out)
                out = self.relu(out)

                # 第二个全连接层
                out = self.dropout(out)
                out = self.fc2(out)

                return out

        model = BasicRNN(input_size, self.hidden_size, output_size)
        return model

    def train_model(self, train_loader, val_loader):
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.001,
            betas=(0.9, 0.999)
        )

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            epochs=self.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )

        print("\nStarting training:")
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                train_loss += loss.item()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    val_loss += criterion(outputs, batch_y).item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], '
                      f'Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}, '
                      f'LR: {scheduler.get_last_lr()[0]:.6f}')

    def prepare_data(self, pose_data, fsr_data, test_size=0.1):

        n_samples = pose_data.shape[0]
        indices = np.random.permutation(n_samples)
        n_test = int(n_samples * test_size)
        n_train = n_samples - n_test

        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        original_shape = pose_data.shape
        pose_reshaped = pose_data.reshape(-1, original_shape[-1])

        scaler_pose = StandardScaler()
        pose_scaled = scaler_pose.fit_transform(pose_reshaped)
        pose_scaled = pose_scaled.reshape(original_shape)

        pose_train_scaled = pose_scaled[train_indices]
        pose_test_scaled = pose_scaled[test_indices]

        scaler_fsr = StandardScaler()
        fsr_train_scaled = scaler_fsr.fit_transform(fsr_data[train_indices])
        fsr_test_scaled = scaler_fsr.transform(fsr_data[test_indices])

        X_train = torch.FloatTensor(pose_train_scaled).to(self.device)
        y_train = torch.FloatTensor(fsr_train_scaled).to(self.device)
        X_test = torch.FloatTensor(pose_test_scaled).to(self.device)
        y_test = torch.FloatTensor(fsr_test_scaled).to(self.device)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

        return train_loader, test_loader, scaler_pose, scaler_fsr

    def process_sequence(self, pose_data, fsr_data):


        train_loader, val_loader, scaler_pose, scaler_fsr = self.prepare_data(pose_data, fsr_data)

        input_size = pose_data.shape[-1]
        output_size = fsr_data.shape[1]
        self.model = self.build_model(input_size, output_size).to(self.device)

        self.train_model(train_loader, val_loader)
        self.model.eval()
        with torch.no_grad():
            pose_reshaped = pose_data.reshape(-1, pose_data.shape[-1])
            pose_scaled = scaler_pose.transform(pose_reshaped)
            pose_scaled = pose_scaled.reshape(pose_data.shape)

            X = torch.FloatTensor(pose_scaled).to(self.device)
            predicted_fsr_scaled = self.model(X).cpu().numpy()
            predicted_fsr = scaler_fsr.inverse_transform(predicted_fsr_scaled)

            return predicted_fsr

    def evaluate_model(self, true_fsr, predicted_fsr):
        """评估模型性能"""
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

def moving_average(data, window_size, axis=0):  # 添加 axis 参数
    """
    对数据应用移动平均平滑。

    Args:
        data: NumPy 数组，模型的输出数据。
        window_size: 移动平均窗口的大小。
        axis:  进行平滑的轴。默认为 0。

    Returns:
        NumPy 数组，平滑后的数据。
    """
    if window_size <= 0:
        raise ValueError("Window size must be positive.")

    if window_size == 1:
        return data  # 没有平滑

    # 使用 'same'
    smoothed_data = np.apply_along_axis(lambda m: np.convolve(m, np.ones(window_size), 'same') / window_size,axis, data)

    return smoothed_data


if __name__ == '__main__':

    # 设置字体以支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 文件路径
    pose_csv_path = "../datasets/pose3.csv"
    fsr_csv_path = "../datasets/downsampled_insole_data4.csv"

    # 加载姿态数据
    pose_data = load_and_preprocess_pose_data(pose_csv_path)

    # 读取和预处理FSR数据
    timestamps, fsr_values = load_fsr_data(fsr_csv_path)
    processed_fsr = preprocess_fsr_data(fsr_values)


    # 确保数据长度匹配，选择较小的长度
    num_samples = min(pose_data.shape[0], processed_fsr.shape[0])
    pose_data = pose_data[:num_samples]
    processed_fsr = processed_fsr[:num_samples]

    # 创建预测器并训练
    predictor = PyTorchFSRPredictor(
        hidden_size=32,  # 更小的隐藏层大小
        learning_rate=0.01,
        batch_size=64,
        num_epochs=100
    )

    # 训练模型并预测FSR值
    predicted_fsr = predictor.process_sequence(pose_data, processed_fsr)

    # 评估模型性能
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