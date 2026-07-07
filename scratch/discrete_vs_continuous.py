"""
scratch/discrete_vs_continuous.py — Speed benchmark: Continuous vs. Discrete Collapse

Compares the computation speed of:
1. Continuous Collapse: Float math (Analytical Cosine Gradient on 256-d vectors)
2. Discrete Collapse: Integer matrix multiplication (5-d Hyper-cube rotations)
"""

import time
import numpy as np
import torch
import torch.nn.functional as F

def benchmark_continuous(steps=100000):
    device = torch.device("cpu")
    # Simulate 256-d float vectors (Standard Pure collapse state & target)
    h = torch.randn(1, 256, device=device)
    target = torch.randn(1, 256, device=device)
    target = F.normalize(target, dim=-1)
    
    strength = torch.tensor([0.5], device=device)
    
    t0 = time.time()
    for _ in range(steps):
        h_norm = h.norm(dim=-1, keepdim=True)
        h_n = h / (h_norm + 1e-8)
        align = (h_n * target).sum(-1, keepdim=True)
        # Gradient update step
        grad = -(target - h_n * align) / (h_norm + 1e-8)
        h = h - strength * grad
        
    dt = time.time() - t0
    return steps / dt

def benchmark_discrete(steps=100000):
    # Simulate a 5-d state vector and an orthogonal rotation matrix
    h = np.array([0, 0, 0, 0, 1])
    # Rotates axes 0 and 1
    R = np.eye(5, dtype=int)
    R[0, 0] = 0
    R[1, 1] = 0
    R[0, 1] = -1
    R[1, 0] = 1
    
    t0 = time.time()
    for _ in range(steps):
        # Discrete group rotation step
        h = R @ h
        
    dt = time.time() - t0
    return steps / dt

def main():
    steps = 200000
    print(f"Running speed comparison over {steps:,} steps on CPU...")
    
    ips_cont = benchmark_continuous(steps)
    print(f"Continuous Collapse Step: {ips_cont:,.2f} ops/sec")
    
    ips_disc = benchmark_discrete(steps)
    print(f"Discrete Group Step:     {ips_disc:,.2f} ops/sec")
    
    speedup = ips_disc / ips_cont
    print("\n==================================================")
    print(f"Verdict: Discrete Group Collapse is {speedup:.1f}x FASTER!")
    print("==================================================")

if __name__ == "__main__":
    main()
