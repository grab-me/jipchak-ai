import numpy as np
import os
import sys
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from three_jaw_grasp import GraspCandidate, FeatureExtractor, BaseGraspAdapter

def collect_from_raw(
    raw_samples: List[Tuple[Any, int]],  # (raw_output, success_label) 리스트
    adapter: BaseGraspAdapter,
    extractor: FeatureExtractor,
    depths: List[np.ndarray],
    output_file: str
):
    """
    외부 모델의 raw output과 성공 여부를 결합하여 학습용 특징 데이터를 생성 및 저장.
    """
    all_features = []
    all_labels = []
    
    for i, (raw, label) in enumerate(raw_samples):
        candidates = adapter.adapt(raw)
        depth = depths[i] if i < len(depths) else None
        
        if not candidates:
            continue
            
        features = extractor.extract_batch(candidates, depth)
        all_features.append(features)
        # 모든 후보가 같은 label을 가진다고 가정하거나, 
        # 특정 후보에 대해서만 레이블링이 되어있어야 함 (여기선 단순화 예시)
        all_labels.append(np.full((len(candidates),), label))
        
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez(output_file, features=X, labels=y)
    print(f"Data collection complete. Saved {len(X)} samples to {output_file}")

if __name__ == "__main__":
    print("Data collection script initialized.")
