import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader


class DataAugmentation:
    def __init__(self, device, noise_factor=0.01, mask_ratio=0.1, scale_range=(0.95, 1.05),
                 time_warp_window=5, rotation_max=5):
        self.device = device
        self.noise_factor = noise_factor
        self.mask_ratio = mask_ratio
        self.scale_range = scale_range
        self.time_warp_window = time_warp_window
        self.rotation_max = rotation_max

    def add_noise(self, sequence):
        """添加高斯噪声"""
        noise = torch.randn_like(sequence) * self.noise_factor
        return sequence + noise

    def random_masking(self, sequence):
        """随机遮蔽部分数据"""
        mask = torch.rand_like(sequence) > self.mask_ratio
        return sequence * mask.float()

    def scaling(self, sequence):
        """随机缩放"""
        scale_factor = torch.empty(1).uniform_(*self.scale_range).to(self.device)
        return sequence * scale_factor

    def time_warping(self, sequence):
        """时间扭曲"""
        batch_size, seq_len, feat_dim = sequence.shape
        warped_sequence = sequence.clone()

        for b in range(batch_size):
            orig_steps = np.arange(seq_len)
            warp_steps = orig_steps + np.random.normal(loc=0, scale=self.time_warp_window,
                                                       size=orig_steps.shape)
            warp_steps = np.sort(warp_steps)

            # 对每个特征维度进行时间扭曲
            for f in range(feat_dim):
                seq = sequence[b, :, f].cpu().numpy()
                warped_seq = np.interp(orig_steps, warp_steps, seq)
                warped_sequence[b, :, f] = torch.from_numpy(warped_seq).float().to(self.device)

        return warped_sequence

    def rotate_pose(self, pose_sequence):
        """对姿态数据进行微小旋转（仅适用于姿态数据）"""
        angle = np.random.uniform(-self.rotation_max, self.rotation_max)
        angle = angle * np.pi / 180  # 转换为弧度

        rot_matrix = torch.tensor([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ], dtype=torch.float32).to(self.device)

        # 假设pose数据是以对的形式组织的（x,y坐标）
        batch_size, seq_len, feat_dim = pose_sequence.shape
        rotated_sequence = pose_sequence.clone()

        for b in range(batch_size):
            for t in range(seq_len):
                for j in range(0, feat_dim, 2):
                    point = pose_sequence[b, t, j:j + 2]
                    rotated_point = torch.matmul(rot_matrix, point)
                    rotated_sequence[b, t, j:j + 2] = rotated_point

        return rotated_sequence

    def __call__(self, sequence, augmentation_types=None):
        """
        应用多种数据增强方法
        augmentation_types: 列表，指定要应用的增强方法
        """
        if augmentation_types is None:
            augmentation_types = ['noise', 'mask', 'scale', 'warp']

        augmented_sequence = sequence.clone()

        for aug_type in augmentation_types:
            if aug_type == 'noise':
                augmented_sequence = self.add_noise(augmented_sequence)
            elif aug_type == 'mask':
                augmented_sequence = self.random_masking(augmented_sequence)
            elif aug_type == 'scale':
                augmented_sequence = self.scaling(augmented_sequence)
            elif aug_type == 'warp':
                augmented_sequence = self.time_warping(augmented_sequence)
            elif aug_type == 'rotate':
                augmented_sequence = self.rotate_pose(augmented_sequence)

        return augmented_sequence


