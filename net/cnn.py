import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

from caculate.fsr_processor import load_fsr_data


def preprocess_fsr_data(fsr_values, smooth_window=5):
    """
    预处理FSR数据

    Parameters:
    fsr_values: 原始FSR数据
    smooth_window: 平滑窗口大小

    Returns:
    processed_values: 归一化后的FSR数据 (NumPy数组)
    scaler: 归一化的MinMaxScaler对象
    """
    # 移动平均平滑
    smoothed_values = pd.DataFrame(fsr_values).rolling(
        window=smooth_window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values

    # 初始化 MinMaxScaler
    scaler = MinMaxScaler()

    # 使用原始数据进行 fit
    scaler.fit(fsr_values)  # 这里确保 scaler 的范围是基于原始数据的范围

    # 对平滑后的数据进行归一化
    processed_values = scaler.transform(smoothed_values)

    # 确保返回的是 NumPy 数组
    if isinstance(processed_values, pd.DataFrame):
        processed_values = processed_values.values
    elif isinstance(processed_values, tuple):
        processed_values = np.array(processed_values)

    return processed_values, scaler

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

def create_sliding_window_dataset(X, y, window_size):
    """创建滑动窗口数据集"""
    X_windows = []
    y_windows = []

    for i in range(len(X) - window_size + 1):
        window = X[i:i+window_size]
        # 正确reshape窗口数据： (window_size, num_channels, num_features) -> (num_channels, window_size * num_features)
        reshaped_window = window.reshape(window_size, *X.shape[1:]) # 保持通道数不变
        reshaped_window = reshaped_window.transpose(1, 0, 2)  # 将时间维度和通道维度交换
        reshaped_window = reshaped_window.reshape(reshaped_window.shape[0], -1)  # 将时间和特征维度合并
        X_windows.append(reshaped_window)
        y_windows.append(y[i+window_size-1])

    return np.array(X_windows), np.array(y_windows)




class PyTorchFSRPredictor:
    def __init__(self, hidden_sizes, learning_rate=0.001, batch_size=32, num_epochs=100, window_size=5):
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = None
        self.device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu")
        self.window_size = window_size

    def build_model(self, input_shape, output_size):
        in_channels, length = input_shape
        model = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(256 * (length // 8), 512),  # Adjust based on pooling layers
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_size)
        )
        return model

    def prepare_data(self, pose_data, fsr_data, test_size=0.1):

        # 展平 pose 数据以用于标准化
        pose_data_2d = pose_data.reshape(pose_data.shape[0], -1)

        # 1. 先划分训练集和测试集的索引
        n_samples = pose_data.shape[0]
        n_test = int(n_samples * test_size)
        n_train = n_samples - n_test
        train_indices = range(n_train)  # 训练集索引
        test_indices = range(n_train, n_samples)  # 测试集索引

        # 2. 只使用训练数据拟合 scaler
        scaler_pose = StandardScaler()
        pose_train_scaled = scaler_pose.fit_transform(pose_data_2d[train_indices])  # 只用训练数据拟合
        pose_test_scaled = scaler_pose.transform(pose_data_2d[test_indices])  # 用训练集的scaler转换测试数据

        scaler_fsr = StandardScaler()
        fsr_train_scaled = scaler_fsr.fit_transform(fsr_data[train_indices])
        fsr_test_scaled = scaler_fsr.transform(fsr_data[test_indices])

        # 将 pose 数据转换回原始形状
        pose_train_scaled = pose_train_scaled.reshape(n_train, *pose_data.shape[1:])
        pose_test_scaled = pose_test_scaled.reshape(n_test, *pose_data.shape[1:])

        # 4. 创建滑动窗口数据集
        X_train, y_train = create_sliding_window_dataset(pose_train_scaled, fsr_train_scaled, self.window_size)
        X_test, y_test = create_sliding_window_dataset(pose_test_scaled, fsr_test_scaled, self.window_size)

        # 3. 转换为张量
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.FloatTensor(y_test).to(self.device)

        # 创建数据集
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)  # 创建测试数据集

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)  # 创建测试数据加载器

        return train_loader, test_loader, scaler_pose, scaler_fsr


    def train_model(self, train_loader, val_loader):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        for epoch in range(self.num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # 验证阶段
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    val_loss += criterion(outputs, batch_y).item()

            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)

            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{self.num_epochs}], '
                      f'Train Loss: {train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}')

    def process_sequence(self, pose_data, fsr_data, scaler_pose, scaler_fsr):
        print(f"Pose data shape: {pose_data.shape}")
        print(f"Window size: {self.window_size}")

        X, _ = create_sliding_window_dataset(pose_data, fsr_data, self.window_size)
        input_shape = X.shape[1:]
        output_size = fsr_data.shape[1]

        if self.model is None:
            self.model = self.build_model(input_shape, output_size).to(self.device)
            train_loader, val_loader, _, _ = self.prepare_data(pose_data, fsr_data)
            self.train_model(train_loader, val_loader)

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(len(pose_data) - self.window_size + 1):
                window = pose_data[i:i + self.window_size]

                # 对窗口内每一帧数据进行缩放
                scaled_window = []
                for frame in window:
                    frame_reshaped = frame.reshape(1, -1)
                    frame_scaled = scaler_pose.transform(frame_reshaped)
                    frame_scaled = frame_scaled.reshape(1, *frame.shape)  # 恢复原始形状
                    scaled_window.append(frame_scaled.squeeze())  # 去除多余维度
                scaled_window = np.array(scaled_window)

                # 转换回滑动窗口格式用于模型输入
                scaled_window = scaled_window.reshape(1, pose_data.shape[1], self.window_size * pose_data.shape[2])
                window_scaled_tensor = torch.FloatTensor(scaled_window).to(self.device)

                output = self.model(window_scaled_tensor)
                predictions.append(output.cpu().numpy().squeeze())

        predictions = np.array(predictions)
        predicted_fsr = scaler_fsr.inverse_transform(predictions)
        predicted_fsr_smoothed = moving_average(predicted_fsr, window_size=10, axis=0)
        return predicted_fsr_smoothed, self.window_size - 1



    def evaluate_model(self, true_fsr, predicted_fsr):
        mse = mean_squared_error(true_fsr, predicted_fsr)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_fsr, predicted_fsr)
        r2 = r2_score(true_fsr, predicted_fsr)
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
    pose_csv_path = "../datasets/pose2.csv"
    fsr_csv_path = "../datasets/downsampled_insole_data4.csv"

    # 加载姿态数据
    pose_data = load_and_preprocess_pose_data(pose_csv_path)

    # 读取和预处理FSR数据
    timestamps, fsr_values = load_fsr_data(fsr_csv_path)
    processed_fsr = preprocess_fsr_data(fsr_values)
    # print(type(pose_data))  # 应该是 <class 'numpy.ndarray'>
    # print(type(processed_fsr))  # 应该是 <class 'numpy.ndarray'>
    processed_fsr = np.array(processed_fsr)
    # 确保数据长度匹配，选择较小的长度
    num_samples = min(pose_data.shape[0], processed_fsr.shape[0])
    pose_data = pose_data[:num_samples]
    processed_fsr = processed_fsr[:num_samples]

    # 创建预测器
    predictor = PyTorchFSRPredictor(
        hidden_sizes=[128, 64],
        learning_rate=0.0001,
        batch_size=128,
        num_epochs=100,
        window_size=10
    )

    # 准备数据并获取scaler
    train_loader, val_loader, scaler_pose, scaler_fsr = predictor.prepare_data(pose_data, processed_fsr)

    # 获取输入形状和输出大小（在滑动窗口之后） - 为了创建模型
    X, _ = create_sliding_window_dataset(pose_data, processed_fsr, predictor.window_size)
    input_shape = X.shape[1:]
    output_size = processed_fsr.shape[1]

    # 创建模型
    predictor.model = predictor.build_model(input_shape, output_size).to(predictor.device)

    # 训练模型
    predictor.train_model(train_loader, val_loader)

    predicted_fsr, start_index = predictor.process_sequence(pose_data, processed_fsr, scaler_pose, scaler_fsr)

    # 调整 true_fsr 的切片，使其长度与 predicted_fsr 相同
    end_index = len(processed_fsr) - start_index - (predictor.window_size - 1)  # calculate the correct end index
    true_fsr_sliced = processed_fsr[start_index:start_index + len(predicted_fsr)]

    mse, rmse, mae, r2 = predictor.evaluate_model(true_fsr_sliced, predicted_fsr)
    print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R-squared: {r2}")



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