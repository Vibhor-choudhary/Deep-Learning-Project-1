import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from models import MLP, CompactCNN
def test_model_shapes_and_parameters():
    x=torch.randn(4,1,32,32)
    assert MLP()(x).shape==(4,46)
    assert CompactCNN()(x).shape==(4,46)
