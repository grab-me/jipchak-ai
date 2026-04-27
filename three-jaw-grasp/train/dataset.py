import torch
from torch.utils.data import Dataset
import numpy as np

class GraspDataset(Dataset):
    """
    평가 모델(MLP) 학습을 위한 데이터셋.
    (features, label) 쌍을 반환.
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float().unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
