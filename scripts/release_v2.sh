#!/bin/bash
# Run this from the livnium-main directory to commit and tag v2.0
# Usage: bash release_v2.sh

set -e

echo "=== Livnium v2.0 Release ==="

# Stage updated README
git add README.md

# Commit
git commit -m "$(cat <<'EOF'
v2.0: Add Three Laws of Livnium + equation of motion

- Add Three Laws section to README:
  Law 1: h₀ = v_h − v_p (relational state formation)
  Law 2: V(h) = −logsumexp(β·cos(h, anchors)) (energy landscape)
  Law 3: h_{t+1} = h_t − α∇V(h_t) (gradient collapse dynamics)
- Empirical equation of motion: trained δ_θ approximates ∇V,
  analytical grad-V matches accuracy and outperforms on neutral (+1.3pp)
- Basin stability: correct predictions 3.9× more stable under perturbation
- Speed: Livnium-native 142× fewer params, 5.3× faster full pipeline
- Laws 2 and 3 were not designed — recovered empirically from trained system
EOF
)"

# Tag the release
git tag -a v2.0 -m "v2.0 — Equation of Motion + Three Laws of Livnium

Key discoveries:
- Empirical equation of motion: h_{t+1} = h_t − α∇V(h_t)
  V(h) = −logsumexp(β·cos(h, anchors)), β=1.0, α=0.2
- Gradient descent on logsumexp cosine energy matches/exceeds trained MLP
- Basin stability as correctness signal (3.9× flip ratio)
- Livnium-native encoder: 772K params, ~80%+ dev, 5.3× faster than BERT
- Three laws fully specify the system (state formation, energy, dynamics)"

echo ""
echo "Commit and tag v2.0 created."
echo "Run: git push origin main && git push origin v2.0"
echo "Then create a GitHub release at: https://github.com/chetanxpatil/livnium/releases/new"
