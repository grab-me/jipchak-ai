import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from three_jaw_grasp.evaluator import GraspScoreMLP
from dataset import GraspDataset

def train(
    data_path: str,
    output_path: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001
):
    # 데이터 로드 (npz 포맷 가정)
    data = np.load(data_path)
    X = data['features']
    y = data['labels']
    
    dataset = GraspDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = GraspScoreMLP(input_dim=X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")
            
    # 가중치 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Training complete. Weights saved to {output_path}")

if __name__ == "__main__":
    # 사용 예시:
    # train("data/processed_features.npz", "weights/three_jaw_mlp.pth")
    print("Train script initialized. Please provide data_path to start training.")
