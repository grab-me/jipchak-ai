"""
기존 MLP 기반 학습 로직 모듈입니다. 파이프라인의 핵심 평가망(ThreeJawEvaluator)을 학습합니다.
"""
import os
import numpy as np
from three_jaw_grasp.factory import TrainerFactory

@TrainerFactory.register("mlp")
def train_mlp_model(data_path: str, output_path: str, epochs: int = 50, batch_size: int = 32, lr: float = 0.001):
    """기존의 GraspScoreMLP를 학습시키는 로직"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    
    from three_jaw_grasp.evaluator import GraspScoreMLP
    from train.dataset import GraspDataset
    
    print("[Trainer: MLP] 핵심 평가망 학습 시작...")
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
            
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"[Trainer: MLP] 학습 완료 및 저장 -> {output_path}")
