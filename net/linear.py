import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from caculate.evaluate_model import evaluate_model_performance, print_evaluation_results, plot_bland_altman, \
    plot_time_series_comparison, plot_correlation_scatter
from caculate.fsr_processor import load_fsr_data, preprocess_fsr_data


# 自定义数据集类
class GRFDataset(Dataset):
    def __init__(self, fsr_data, grf_data):
        self.fsr_data = torch.FloatTensor(fsr_data)
        self.grf_data = torch.FloatTensor(np.mean(grf_data, axis=1))  # (frames, 3)

    def __len__(self):
        return len(self.fsr_data)

    def __getitem__(self, idx):
        return self.fsr_data[idx], self.grf_data[idx]


# 定义神经网络模型
class GRFNet(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64, 32]):
        super(GRFNet, self).__init__()

        layers = []
        prev_size = input_size

        # 构建隐藏层
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size

        # 输出层
        layers.append(nn.Linear(prev_size, 3))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PyTorchGRFProcessor:
    def __init__(self, hidden_sizes=[64, 32], learning_rate=0.001, batch_size=32,
                 num_epochs=100, device=None):
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu")

        self.model = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        # 用于记录训练过程
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader, criterion, optimizer):
        self.model.train()
        total_loss = 0

        for fsr, grf in train_loader:
            fsr, grf = fsr.to(self.device), grf.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(fsr)
            loss = criterion(outputs, grf)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for fsr, grf in val_loader:
                fsr, grf = fsr.to(self.device), grf.to(self.device)
                outputs = self.model(fsr)
                loss = criterion(outputs, grf)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def process_sequence(self, fsr_data, grf_data):
        # 数据预处理
        X = self.scaler_x.fit_transform(fsr_data)
        y = self.scaler_y.fit_transform(np.mean(grf_data, axis=1))

        # 划分数据集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # 创建数据加载器
        train_dataset = GRFDataset(X_train, y_train.reshape(-1, 1, 3))
        val_dataset = GRFDataset(X_val, y_val.reshape(-1, 1, 3))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # 初始化模型
        self.model = GRFNet(input_size=fsr_data.shape[1],
                            hidden_sizes=self.hidden_sizes).to(self.device)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

        # 早停设置
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_model_state = None

        # 训练循环
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            val_loss = self.validate(val_loader, criterion)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            scheduler.step(val_loss)

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], '
                      f'Train Loss: {train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}')

        # 加载最佳模型
        self.model.load_state_dict(best_model_state)

        # 预测
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(self.scaler_x.transform(fsr_data)).to(self.device)
            predicted = self.model(X_tensor).cpu().numpy()
            estimated_grf = self.scaler_y.inverse_transform(predicted)

        # 绘制训练过程
        self.plot_training_history()

        return None, estimated_grf

    def plot_training_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_history.png')
        plt.close()


def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 设置字体以支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 文件路径
    csv_file_path = "../datasets/downsampled_insole_data4.csv"

    # 读取和预处理数据
    timestamps, fsr_values = load_fsr_data(csv_file_path)
    processed_fsr = preprocess_fsr_data(fsr_values)

    # 加载GRF数据
    data = np.load("../output/frame_output.npz")
    grf_data = data['pred_grf']

    if grf_data.ndim == 4:
        grf_data = grf_data[0]

    # 确保数据长度匹配
    min_frames = min(len(processed_fsr), len(grf_data))
    processed_fsr = processed_fsr[:min_frames]
    grf_data = grf_data[:min_frames]

    # 创建并训练模型
    processor = PyTorchGRFProcessor(
        hidden_sizes=[64, 32],
        learning_rate=0.001,
        batch_size=32,
        num_epochs=100
    )

    # 处理数据序列
    calibration_matrix, estimated_grf = processor.process_sequence(processed_fsr, grf_data)

    # 评估模型性能
    metrics = evaluate_model_performance(estimated_grf, grf_data)
    print_evaluation_results(metrics)

    # 绘制Bland-Altman图和其他可视化
    plot_bland_altman(estimated_grf, grf_data)
    plot_time_series_comparison(estimated_grf, grf_data)
    plot_correlation_scatter(estimated_grf, grf_data)

    # 计算并打印整体性能指标
    overall_r2 = np.mean([metrics['R2'][d] for d in ['X', 'Y', 'Z']])
    overall_nrmse = np.mean([metrics['NRMSE'][d] for d in ['X', 'Y', 'Z']])

    print("\n整体性能指标:")
    print(f"平均 R²: {overall_r2:.4f}")
    print(f"平均 NRMSE: {overall_nrmse:.4f}")


if __name__ == "__main__":
    main()