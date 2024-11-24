import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, Subset
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from caculate.fsr_processor import load_fsr_data, preprocess_fsr_data


class PressNetSimple(nn.Module):
    def __init__(self, input_size, output_size, N=4, W=256):
        super(PressNetSimple, self).__init__()
        self.N = N
        self.W = W

        # 计算展平后的输入大小 (33 个关键点 * 3 个坐标)
        self.flattened_input_size = np.prod(input_size)

        # 可以添加一些特征提取层
        self.feature_extraction = nn.Sequential(
            nn.Linear(self.flattened_input_size, W * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(W * 2, W)
        )

        # 中间层
        self.hidden_layers = nn.ModuleList([
            nn.Linear(W, W) for _ in range(N - 1)
        ])

        # 输出层
        self.output_layer = nn.Linear(W, output_size)

        # Dropout层
        self.dropout = nn.Dropout(0.5)

        # 激活函数
        self.activation = nn.ReLU()

    def forward(self, x):
        # 输入层
        x = x.view(-1, self.flattened_input_size)
        x = self.feature_extraction(x)

        # 中间层（带残差连接）
        for layer in self.hidden_layers:
            identity = x
            x = self.activation(layer(x))
            x = self.dropout(x)
            x = x + identity  # 残差连接

        # 输出层
        x = self.output_layer(x)
        return x



def load_pose_data(csv_file_path):
    """
    读取姿态数据
    """
    # 读取CSV文件，跳过第一行（列名）
    df = pd.read_csv(csv_file_path, low_memory=False)

    # 只选择包含坐标的列（x, y, z），排除visibility列和其他列
    coord_columns = [col for col in df.columns if any(x in col for x in ['_x', '_y', '_z'])
                     and 'visibility' not in col]

    # 删除第一行（如果它包含列名）
    if df.iloc[0].equals(pd.Series(df.columns)):
        df = df.iloc[1:]

    # 将数据转换为浮点数
    pose_data = df[coord_columns].astype(float).values

    # 重塑数据为 (samples, 33, 3) 的形状
    num_samples = pose_data.shape[0]
    pose_data = pose_data.reshape(num_samples, -1, 3)

    return pose_data



class PyTorchFSRPredictor:
    def __init__(self, hidden_sizes, learning_rate=0.001, batch_size=32, num_epochs=100,window_size=5):
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = None
        self.device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu")
        self.window_size = window_size

    def build_model(self, input_shape, output_size):
        return PressNetSimple(
            input_size=input_shape,
            output_size=output_size,
            N=4,  # 网络深度
            W=1024  # 网络宽度
        )



    def prepare_data(self, grf_data, fsr_data, test_size=0.2):

        # 展平 GRF 数据以用于标准化
        grf_data_2d = grf_data.reshape(grf_data.shape[0], -1)

        # 1. 先划分训练集和测试集的索引
        n_samples = grf_data.shape[0]
        n_test = int(n_samples * test_size)
        n_train = n_samples - n_test
        train_indices = range(n_train)  # 训练集索引
        test_indices = range(n_train, n_samples)  # 测试集索引

        # 2. 只使用训练数据拟合 scaler
        scaler_grf = StandardScaler()
        grf_train_scaled = scaler_grf.fit_transform(grf_data_2d[train_indices])  # 只用训练数据拟合
        grf_test_scaled = scaler_grf.transform(grf_data_2d[test_indices])  # 用训练集的scaler转换测试数据

        scaler_fsr = StandardScaler()
        fsr_train_scaled = scaler_fsr.fit_transform(fsr_data[train_indices])
        fsr_test_scaled = scaler_fsr.transform(fsr_data[test_indices])

        # 将 GRF 数据转换回原始形状
        grf_train_scaled = grf_train_scaled.reshape(n_train, *grf_data.shape[1:])
        grf_test_scaled = grf_test_scaled.reshape(n_test, *grf_data.shape[1:])

        # 3. 转换为张量
        X_train = torch.FloatTensor(grf_train_scaled).to(self.device)
        y_train = torch.FloatTensor(fsr_train_scaled).to(self.device)
        X_test = torch.FloatTensor(grf_test_scaled).to(self.device)
        y_test = torch.FloatTensor(fsr_test_scaled).to(self.device)

        # 创建数据集
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)  # 创建测试数据集

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)  # 创建测试数据加载器

        return train_loader, test_loader, scaler_grf, scaler_fsr

    # def prepare_data(self, grf_data, fsr_data):
    #
    #     # 展平 GRF 数据以用于标准化
    #     grf_data_2d = grf_data.reshape(grf_data.shape[0], -1)  # (430, 3 * 29)
    #
    #     # 标准化 GRF 和 FSR 数据
    #     scaler_grf = StandardScaler()
    #     grf_data_scaled = scaler_grf.fit_transform(grf_data_2d)  # 使用展平后的数据
    #
    #     scaler_fsr = StandardScaler()
    #     fsr_data_scaled = scaler_fsr.fit_transform(fsr_data)
    #
    #     # 将 GRF 数据转换回原始形状
    #     grf_data_scaled = grf_data_scaled.reshape(grf_data.shape)  # (430, 3, 29)
    #
    #     # 转换为张量 - 使用标准化后的数据!
    #     X = torch.FloatTensor(grf_data_scaled).to(self.device)
    #     y = torch.FloatTensor(fsr_data_scaled).to(self.device)
    #
    #     # 创建数据集
    #     dataset = TensorDataset(X, y)
    #
    #     # 划分训练集和验证集
    #     train_size = int(0.8 * len(dataset))
    #     val_size = len(dataset) - train_size
    #     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    #
    #     # 创建数据加载器
    #     train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
    #     val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
    #
    #     return train_loader, val_loader, scaler_grf, scaler_fsr

    def train_model(self, train_loader, val_loader):
        criterion = nn.MSELoss()
        # 使用 Huber 损失函数
        # criterion = nn.HuberLoss(delta=0.5)  # 可以调整 delta 参数
        # criterion = nn.L1Loss()  # 使用 MAE 损失函数
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

    def process_sequence(self, grf_data, fsr_data):
        # ... (数据预处理)
        train_loader, val_loader, scaler_grf, scaler_fsr = self.prepare_data(grf_data, fsr_data)
        # Reshape GRF data for Conv1d: (samples, channels, length)
        num_samples = grf_data.shape[0]


        # 构建模型
        input_shape = grf_data.shape[1:]  # 从 grf_data 获取输入形状
        output_size = fsr_data.shape[1]
        self.model = self.build_model(input_shape, output_size).to(self.device)

        # (训练和预测)
        # 训练模型
        self.train_model(train_loader, val_loader)

        # 预测FSR值
        self.model.eval()
        with torch.no_grad():

            # 保持 GRF 数据的原始形状，直接进行标准化
            grf_data_scaled = scaler_grf.transform(grf_data.reshape(grf_data.shape[0], -1)).reshape(grf_data.shape)
            X = torch.FloatTensor(grf_data_scaled).to(self.device)

            predicted_fsr_scaled = self.model(X).cpu().numpy()
            predicted_fsr = scaler_fsr.inverse_transform(predicted_fsr_scaled)
            predicted_fsr_smoothed = moving_average(predicted_fsr, window_size=10, axis=0)  # 指定 axis

            return predicted_fsr_smoothed

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
    csv_file_path = "../datasets/downsampled_insole_data4.csv"
    pose_file_path = "../datasets/pose3.csv"  # 姿态数据文件路径
    # 读取和预处理数据
    timestamps, fsr_values = load_fsr_data(csv_file_path)
    processed_fsr = preprocess_fsr_data(fsr_values)

    # 加载姿态数据
    pose_data = load_pose_data(pose_file_path)

    # 确保FSR数据长度匹配
    num_samples = min(pose_data.shape[0], processed_fsr.shape[0])
    pose_data = pose_data[:num_samples]
    processed_fsr = processed_fsr[:num_samples]



    # ... (创建预测器，训练，评估)
    # 确保数据是float32类型
    predictor = PyTorchFSRPredictor(
        hidden_sizes=[128, 64],  # 这些隐藏层大小现在用于全连接层
        learning_rate=0.001,
        batch_size=64,
        num_epochs=200
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