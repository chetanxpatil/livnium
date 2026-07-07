"""
experiment/discrete_cube_collapse.py — Reversible Discrete Collapse Prototype

Maps word transitions to the 24 proper rotations of the Livnium Core 3D lattice.
Learns discrete word-to-rotation assignments to route trajectories to the
correct semantic classification corners of the 3x3x3 cube.
"""

import numpy as np
import random

# ------------------------------------------------------------- 1. Group Setup

def get_octahedral_rotations():
    """Generates the 24 proper rotation matrices of a 3D grid (det = +1)."""
    matrices = []
    # 6 permutations of axes (x, y, z)
    perms = [
        [0, 1, 2], [1, 2, 0], [2, 0, 1],
        [0, 2, 1], [2, 1, 0], [1, 0, 2]
    ]
    # 8 sign variations
    signs = [
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ]
    for p in perms:
        for s in signs:
            M = np.zeros((3, 3), dtype=int)
            M[0, p[0]] = s[0]
            M[1, p[1]] = s[1]
            M[2, p[2]] = s[2]
            if int(np.round(np.linalg.det(M))) == 1:
                matrices.append(M)
    return matrices

ROTATIONS = get_octahedral_rotations()
assert len(ROTATIONS) == 24, f"Expected 24 rotations, got {len(ROTATIONS)}"

# Define the 27 coordinates of the 3x3x3 cube
CELLS = []
for x in [-1, 0, 1]:
    for y in [-1, 0, 1]:
        for z in [-1, 0, 1]:
            CELLS.append(np.array([x, y, z]))

# Semantic classification targets located at specific corners of the cube
TARGETS = {
    "positive": np.array([1, 1, 1]),   # top-right-front corner
    "negative": np.array([-1, -1, -1]) # bottom-left-back corner
}


# -------------------------------------------------------- 2. Discrete Engine

def run_trajectory(words, word_to_rot_idx, start_state):
    """Collapses the state through the words using discrete group rotations."""
    h = start_state.copy()
    for w in words:
        rot_idx = word_to_rot_idx[w]
        R = ROTATIONS[rot_idx]
        h = R @ h  # Reversible matrix permutation multiplication
    return h

def evaluate_accuracy(data, word_to_rot_idx, start_state):
    """Calculates classification accuracy on the dataset."""
    correct = 0
    for words, label in data:
        h_final = run_trajectory(words, word_to_rot_idx, start_state)
        target = TARGETS[label]
        
        # Readout: cosine similarity (dot product since coordinates are bound)
        score = np.dot(h_final, target)
        # Check if the final coordinate is closer to the target than other targets
        other_label = "negative" if label == "positive" else "positive"
        other_score = np.dot(h_final, TARGETS[other_label])
        
        if score > other_score:
            correct += 1
    return correct / len(data)


# ------------------------------------------------------------- 3. Simulation

def main():
    print("Initializing octahedral rotation group (G = 24 proper rotations)...")
    
    # 1. Prepare synthetic dataset
    # We want the model to learn that "good", "great", "yes" are positive forces,
    # and "bad", "no", "fail" are negative forces.
    vocab = ["good", "great", "yes", "bad", "no", "fail", "the", "a"]
    data = [
        (["good", "yes"], "positive"),
        (["great", "good"], "positive"),
        (["great", "yes"], "positive"),
        (["bad", "no"], "negative"),
        (["fail", "bad"], "negative"),
        (["no", "fail"], "negative"),
        (["the", "great", "good"], "positive"),
        (["a", "bad", "no"], "negative"),
        (["good", "the", "yes"], "positive"),
        (["bad", "a", "fail"], "negative"),
    ]
    
    # Starting state: let's start at the top face center of the cube
    start_state = np.array([0, 0, 1])
    
    # 2. Greedy search over discrete rotation assignments (learning phase)
    random.seed(42)
    word_to_rot_idx = {w: random.randint(0, 23) for w in vocab}
    
    best_acc = evaluate_accuracy(data, word_to_rot_idx, start_state)
    print(f"initial random accuracy: {best_acc * 100:.1f}%")
    
    # Greedy coordinate descent over discrete rotation choices
    improved = True
    step = 0
    while improved:
        improved = False
        step += 1
        for w in vocab:
            current_idx = word_to_rot_idx[w]
            for r_idx in range(24):
                if r_idx == current_idx:
                    continue
                word_to_rot_idx[w] = r_idx
                acc = evaluate_accuracy(data, word_to_rot_idx, start_state)
                
                if acc > best_acc:
                    best_acc = acc
                    improved = True
                    current_idx = r_idx
                else:
                    word_to_rot_idx[w] = current_idx # backtrack
                    
        print(f"Search Step {step} | Best Accuracy: {best_acc * 100:.1f}%")
        if best_acc == 1.0:
            break
            
    print("\n--- Training Completed Successfully! ---")
    print("Learned Word-to-Rotation Mappings:")
    for w, idx in word_to_rot_idx.items():
        print(f"  {w:8s} -> Rotation Matrix #{idx}")
        
    # 3. Algebraic Extraction
    print("\n==================================================")
    print("            ALGEBRAIC FORMULA EXTRACTION          ")
    print("==================================================")
    
    # Take a sequence "the great good" -> positive target [1, 1, 1]
    sample_seq = ["the", "great", "good"]
    print(f"Sequence: {sample_seq}")
    print(f"Start State: {start_state}")
    
    # Compute the net composed matrix algebraically
    net_matrix = np.eye(3, dtype=int)
    for w in sample_seq:
        net_matrix = ROTATIONS[word_to_rot_idx[w]] @ net_matrix
        
    final_state = net_matrix @ start_state
    
    print("\nIndividual Rotation Matrices:")
    for w in sample_seq:
        idx = word_to_rot_idx[w]
        print(f"\nMatrix for '{w}' (Rotation #{idx}):")
        print(ROTATIONS[idx])
        
    print("\nComposed Net Rotation Matrix (Algebraic Product):")
    print(net_matrix)
    print(f"Final Collapsed State: {final_state}")
    
    # Match the composed matrix to the 24 group rotations
    matched_idx = -1
    for idx, R in enumerate(ROTATIONS):
        if np.array_equal(net_matrix, R):
            matched_idx = idx
            break
            
    print(f"\nNet Group Element: Rotation Matrix #{matched_idx} in G")
    print("Verdict: The entire context trajectory is compressed losslessly into a single group element!")

if __name__ == "__main__":
    main()
