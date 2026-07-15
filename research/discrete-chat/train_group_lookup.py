"""
experiment/train_group_lookup.py — Robust Precomputed Group Table Lookup Model

Implements Method 1: No matrix multiplications in the sequence loop.
Sequence composition is computed entirely as a chain of integer lookups
in a precomputed 24x24 multiplication table.
"""

import os
import sys
import random
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../chat-brain"))
from chat_reply import read_pairs

# -------------------------------------------------------- 1. Octahedral Setup

def get_octahedral_rotations():
    """Generates the 24 proper rotation matrices in 3D (det = +1)."""
    matrices = []
    perms = [
        [0, 1, 2], [1, 2, 0], [2, 0, 1],
        [0, 2, 1], [2, 1, 0], [1, 0, 2]
    ]
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

# Build the 24x24 multiplication table: MULT_TABLE[i, j] = k  <=>  R_i @ R_j = R_k
def build_multiplication_table(rotations):
    table = np.zeros((24, 24), dtype=int)
    for i in range(24):
        for j in range(24):
            prod = rotations[i] @ rotations[j]
            # Match product to one of the 24 rotations
            match_idx = -1
            for k in range(24):
                if np.array_equal(prod, rotations[k]):
                    match_idx = k
                    break
            table[i, j] = match_idx
    return table

MULT_TABLE = build_multiplication_table(ROTATIONS)

# Semantic targets (corners of the 3D cube)
TARGETS = {
    "question": np.array([1, 1, 1]),
    "statement": np.array([-1, -1, -1]),
    "greeting": np.array([1, -1, 1])
}

# ------------------------------------------------------------- 2. Data Labeler

def label_sentence(msg):
    m = msg.lower()
    if "?" in m or any(w in m for w in ["what", "why", "how", "who", "when"]):
        return "question"
    elif any(w in m for w in ["hi", "hello", "hey", "bro", "cool"]):
        return "greeting"
    else:
        return "statement"

# ------------------------------------------------------------- 3. Lookup Engine

def run_lookup_trajectory(words, word_to_rot_idx):
    """Composes rotations using ONLY integer table lookups."""
    state_rot_idx = 0  # Start at identity (matrix index 0)
    for w in words:
        if w in word_to_rot_idx:
            rot_w = word_to_rot_idx[w]
            # Chain composition: R_new = R_current @ R_word
            state_rot_idx = MULT_TABLE[state_rot_idx, rot_w]
    return state_rot_idx

def evaluate_accuracy(data, word_to_rot_idx, start_vector):
    correct = 0
    for words, label in data:
        # 1. Get the final net rotation index purely from lookups
        final_rot_idx = run_lookup_trajectory(words, word_to_rot_idx)
        
        # 2. Apply final rotation to start state ONCE
        R_final = ROTATIONS[final_rot_idx]
        h_final = R_final @ start_vector
        
        # 3. Readout classification
        best_label = None
        best_score = float("-inf")
        for lbl, tgt in TARGETS.items():
            score = np.dot(h_final, tgt)
            if score > best_score:
                best_score = score
                best_label = lbl
                
        if best_label == label:
            correct += 1
    return correct / len(data)

# ------------------------------------------------------------- 4. Run & Compare

def main():
    DATA_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "chat-brain", "data", "chat_context.tsv")
    )
    
    # 1. Load and prep data (Train/Test Split for Robustness)
    pairs = read_pairs(DATA_PATH, max_lines=60)
    dataset = []
    vocab_counter = {}
    for msg, _ in pairs:
        label = label_sentence(msg)
        words = msg.split()
        dataset.append((words, label))
        for w in words:
            vocab_counter[w] = vocab_counter.get(w, 0) + 1
            
    # Shrink vocab to frequent terms
    vocab = sorted([w for w, c in vocab_counter.items() if c >= 2], key=lambda w: -vocab_counter[w])[:60]
    
    # Train / Test split (80% / 20%)
    random.seed(42)
    random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.8)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    print(f"Loaded {len(dataset)} sentences. Train: {len(train_data)} | Test: {len(test_data)}")
    print(f"Vocabulary size: {len(vocab)} words.")
    
    # Starting state: top face center [0, 0, 1]
    start_vector = np.array([0, 0, 1])
    
    # Initialize random assignments
    word_to_rot_idx = {w: random.randint(0, 23) for w in vocab}
    
    init_train_acc = evaluate_accuracy(train_data, word_to_rot_idx, start_vector)
    init_test_acc = evaluate_accuracy(test_data, word_to_rot_idx, start_vector)
    
    print(f"\nInitial Train Acc: {init_train_acc*100:.2f}% | Test Acc: {init_test_acc*100:.2f}%")
    
    # 2. Train using discrete coordinate descent
    print("Training word assignments on Group Multiplication Table...")
    best_train_acc = init_train_acc
    improved = True
    step = 0
    
    while improved and step < 5:
        improved = False
        step += 1
        random.shuffle(vocab)
        for w in vocab:
            curr_idx = word_to_rot_idx[w]
            for candidate_idx in range(24):
                if candidate_idx == curr_idx:
                    continue
                word_to_rot_idx[w] = candidate_idx
                acc = evaluate_accuracy(train_data, word_to_rot_idx, start_vector)
                
                if acc > best_train_acc:
                    best_train_acc = acc
                    improved = True
                    curr_idx = candidate_idx
                else:
                    word_to_rot_idx[w] = curr_idx # backtrack
                    
        print(f"  Step {step} | Best Train Accuracy: {best_train_acc*100:.2f}%")
        
    test_acc = evaluate_accuracy(test_data, word_to_rot_idx, start_vector)
    print(f"\n--- Robust Testing Complete ---")
    print(f"Final Train Accuracy: {best_train_acc*100:.2f}%")
    print(f"Final Test Accuracy:  {test_acc*100:.2f}%")
    
    # 3. Benchmark Speed comparison (Matrix Multiplications vs. Integer Lookups)
    print("\n--- Speed Benchmark (100,000 runs) ---")
    sample_sentence = ["good", "great", "yes", "bad", "no", "the", "a"] * 5
    
    # Matrix version benchmark
    t0 = time.time()
    for _ in range(100000):
        h = start_vector.copy()
        for w in sample_sentence:
            if w in word_to_rot_idx:
                idx = word_to_rot_idx[w]
                h = ROTATIONS[idx] @ h
    dt_matrix = time.time() - t0
    print(f"Matrix Multiply version: {100000/dt_matrix:,.2f} runs/sec")
    
    # Precomputed table lookup version benchmark
    t0 = time.time()
    for _ in range(100000):
        idx_rot = 0
        for w in sample_sentence:
            if w in word_to_rot_idx:
                idx_rot = MULT_TABLE[idx_rot, word_to_rot_idx[w]]
        h_final = ROTATIONS[idx_rot] @ start_vector
    dt_lookup = time.time() - t0
    print(f"Table Lookup version:   {100000/dt_lookup:,.2f} runs/sec")
    
    print(f"Speedup from Table Lookup: {dt_matrix/dt_lookup:.2f}x FASTER!")

if __name__ == "__main__":
    main()
