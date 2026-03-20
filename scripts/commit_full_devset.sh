#!/bin/bash
# Run from livnium-main root
git add README.md
git commit -m "Update grad-V result to full dev set (9842 samples)

Full SNLI dev validation of Law 3 (gradient collapse dynamics):
- Accuracy: +0.16pp over trained MLP (82.21% vs 82.05%)
- Neutral recall: +1.02pp (72.18% vs 71.16%)
- δ_θ contribution: only 0.23pp — pure geometry drives dynamics
- Collapse 2.1× faster without MLP
- Result consistent with 2000-sample run (+0.30%)
- Equation of motion confirmed on full dev: h_{t+1}=h_t−α∇V(h_t)"

git push origin main
