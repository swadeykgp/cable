"""Run: python examples/minimal_usage.py"""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cable import CABLE2D

x = torch.randn(2, 32, 64, 64)
operator = CABLE2D(channels=32, groups=4)
y = operator(x)
print("Input shape:", tuple(x.shape))
print("Output shape:", tuple(y.shape))
print("Operator parameters:", sum(p.numel() for p in operator.parameters()))
