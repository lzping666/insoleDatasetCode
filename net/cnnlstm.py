import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from caculate.fsr_processor import load_fsr_data, preprocess_fsr_data


class PyTorchFSRPredictor:
    def __init__(self, hidden_size, num_layers, learning_rate=0.001, batch_size=32, num_epochs=200):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = None
        self.device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu")

    def build_model(self, input_channels, output_size):
        class CNNLSTMModel(nn.Module):
            def __init__(self, input_channels, hidden_size, num_layers, output_size):
                super(CNNLSTMModel, self).__init__()
                self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
                self.pool = nn.MaxPool1d(kernel_size=2, stride=1)  # Adjust stride to avoid size reduction
                self.lstm = nn.LSTM(128, hidden_size, num_layers, batch_first=True, dropout=0.3)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                x = x.transpose(1, 2)  # Swap dimensions for Conv1d
                x = torch.relu(self.conv1(x))
                x = self.pool(x)
                x = torch.relu(self.conv2(x))
                x = self.pool(x)
                x = x.transpose(1, 2)  # Swap back for LSTM
                lstm_out, _ = self.lstm(x)
                out = self.fc(lstm_out[:, -1, :])
                return out

        return CNNLSTMModel(input_channels, self.hidden_size, self.num_layers, output_size)

    def prepare_data(self, grf_data, fsr_data, test_size=0.2):
        grf_data_2d = grf_data.reshape(grf_data.shape[0], -1)
        n_samples = grf_data.shape[0]
        n_test = int(n_samples * test_size)
        n_train = n_samples - n_test

        train_indices = range(n_train)
        test_indices = range(n_train, n_samples)

        scaler_grf = StandardScaler()
        grf_train_scaled = scaler_grf.fit_transform(grf_data_2d[train_indices])
        grf_test_scaled = scaler_grf.transform(grf_data_2d[test_indices])

        scaler_fsr = StandardScaler()
        fsr_train_scaled = scaler_fsr.fit_transform(fsr_data[train_indices])
        fsr_test_scaled = scaler_fsr.transform(fsr_data[test_indices])

        grf_train_scaled = grf_train_scaled.reshape(n_train, *grf_data.shape[1:])
        grf_test_scaled = grf_test_scaled.reshape(n_test, *grf_data.shape[1:])

        X_train = torch.FloatTensor(grf_train_scaled).to(self.device)
        y_train = torch.FloatTensor(fsr_train_scaled).to(self.device)
        X_test = torch.FloatTensor(grf_test_scaled).to(self.device)
        y_test = torch.FloatTensor(fsr_test_scaled).to(self.device)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader, scaler_grf, scaler_fsr

    def train_model(self, train_loader, val_loader):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        best_val_loss = float('inf')
        patience, trials = 10, 0

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    val_loss += criterion(outputs, batch_y).item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trials = 0
            else:
                trials += 1
                if trials >= patience:
                    print("Early stopping on epoch:", epoch)
                    break

            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    def process_sequence(self, grf_data, fsr_data):
        train_loader, val_loader, scaler_grf, scaler_fsr = self.prepare_data(grf_data, fsr_data)
        input_channels = grf_data.shape[2]  # Correctly set input channels
        output_size = fsr_data.shape[1]
        self.model = self.build_model(input_channels, output_size).to(self.device)
        self.train_model(train_loader, val_loader)

        self.model.eval()
        with torch.no_grad():
            grf_data_scaled = scaler_grf.transform(grf_data.reshape(grf_data.shape[0], -1)).reshape(grf_data.shape)
            X = torch.FloatTensor(grf_data_scaled).to(self.device)

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

    csv_file_path = "../datasets/downsampled_insole_data.csv"
    timestamps, fsr_values = load_fsr_data(csv_file_path)
    processed_fsr = preprocess_fsr_data(fsr_values)

    data = np.load("../output/frame_output.npz")
    grf_data = data['pred_grf']
    grf_data = grf_data.squeeze(0).transpose(0, 2, 1)

    num_samples = grf_data.shape[0]
    processed_fsr = processed_fsr[:num_samples]

    predictor = PyTorchFSRPredictor(
        hidden_size=64,
        num_layers=2,
        learning_rate=0.001,
        batch_size=32,
        num_epochs=200
    )

    predicted_fsr = predictor.process_sequence(grf_data, processed_fsr)
    mse, rmse, mae, r2 = predictor.evaluate_model(processed_fsr, predicted_fsr)

    plt.figure(figsize=(12, 6))
    plt.plot(processed_fsr, label='True FSR')
    plt.plot(predicted_fsr, label='Predicted FSR')
    plt.legend()
    plt.title('FSR Prediction Results')
    plt.xlabel('Time steps')
    plt.ylabel('FSR values')
    plt.show()