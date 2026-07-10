"""
experiment/test_discrete_chat.py — Hyper-Livnium Core Chat Classifier

Trains a 5-dimensional discrete rotation model (243 grid states)
on your real conversation context pairs.
"""

import os
import sys
import random
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../chat"))
from chat_reply import read_pairs
DEFAULT_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "../chat/data/chat_context.tsv"))

# ----------------------------------------------------------- 1. 5D Group Setup

DIM = 5  # 5-dimensional Hyper-Livnium Core (3^5 = 243 discrete states)

def generate_generators(dim):
    """Generates a set of discrete rotation generators (axes permutations and sign flips)

    For D=5, we define 10 basic orthogonal rotation matrices as actions.
    """
    generators = []
    # 1. Identity (no-op)
    generators.append(np.eye(dim, dtype=int))
    
    # 2. Simple coordinate sign flips (flips along axis i)
    for i in range(dim):
        M = np.eye(dim, dtype=int)
        M[i, i] = -1
        generators.append(M)
        
    # 3. Simple axis swaps (rotate plane i-j by 90 degrees)
    for i in range(dim - 1):
        M = np.eye(dim, dtype=int)
        M[i, i] = 0
        M[i+1, i+1] = 0
        M[i, i+1] = -1
        M[i+1, i] = 1
        generators.append(M)
        
    return generators

GENERATORS = generate_generators(DIM)
N_ACTIONS = len(GENERATORS)

# Target wells: project 3 distinct semantic output wells onto the 5D corners
TARGETS = {
    "question": np.array([1, 1, 1, 0, 0]),
    "statement": np.array([-1, -1, -1, 0, 0]),
    "greeting": np.array([0, 0, 0, 1, 1])
}


# ------------------------------------------------------------- 2. Data Prep

def label_sentence(msg):
    """Categorizes the real message into statement, question, or greeting."""
    m = msg.lower()
    if "?" in m or any(w in m for w in ["what", "why", "how", "who", "when"]):
        return "question"
    elif any(w in m for w in ["hi", "hello", "hey", "bro", "cool"]):
        return "greeting"
    else:
        return "statement"


# ------------------------------------------------------------- 3. Engine

def run_trajectory(words, word_to_action, start_state):
    assert start_state.ndim == 1 and start_state.shape[0] == DIM, f"start_state must be 1D of size {DIM}, got shape {start_state.shape}"
    h = start_state.copy()
    for w in words:
        if w in word_to_action:
            action_idx = word_to_action[w]
            assert 0 <= action_idx < len(GENERATORS), f"Action index {action_idx} out of range [0, {len(GENERATORS) - 1}]"
            R = GENERATORS[action_idx]
            h = R @ h
    return h

def evaluate_accuracy(data, word_to_action, start_state):
    correct = 0
    for words, label in data:
        h_final = run_trajectory(words, word_to_action, start_state)
        
        # Calculate dot product score against the three target wells
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


# ------------------------------------------------------------- 4. Run

def main():
    print(f"Initializing {DIM}D Hyper-Livnium Core...")
    print(f"Number of discrete actions: {N_ACTIONS}")
    
    # Load real conversation contexts
    pairs = read_pairs(DEFAULT_DATA, max_lines=40)
    print(f"Loaded {len(pairs):,} contexts from database.")
    
    # Process contexts into token sequences and labels
    dataset = []
    vocab_counter = {}
    for msg, _ in pairs:
        label = label_sentence(msg)
        words = msg.split()
        dataset.append((words, label))
        for w in words:
            vocab_counter[w] = vocab_counter.get(w, 0) + 1
            
    # Shrink vocabulary to frequent words to keep optimization fast
    vocab = sorted([w for w, c in vocab_counter.items() if c >= 2], key=lambda w: -vocab_counter[w])[:60]
    print(f"Vocab size (capped top 60): {len(vocab)} words.")
    print(f"Class distribution: "
          f"Questions: {sum(1 for _, l in dataset if l=='question')} | "
          f"Greetings: {sum(1 for _, l in dataset if l=='greeting')} | "
          f"Statements: {sum(1 for _, l in dataset if l=='statement')}")
    
    # Start state: centered on the hypercube axis
    start_state = np.array([0, 0, 0, 0, 1])
    
    # Initialize random actions
    random.seed(42)
    word_to_action = {w: random.randint(0, N_ACTIONS - 1) for w in vocab}
    
    initial_acc = evaluate_accuracy(dataset, word_to_action, start_state)
    print(f"\nInitial random classification accuracy: {initial_acc * 100:.2f}%")
    
    # Run discrete search (hill-climbing) over actions
    print("Training word-to-rotation assignments via discrete search...")
    best_acc = initial_acc
    improved = True
    step = 0
    
    t0 = time.time()
    while improved and step < 5:
        improved = False
        step += 1
        # Shuffle vocab search order to escape local minima
        random.shuffle(vocab)
        for w in vocab:
            current_action = word_to_action[w]
            for action_idx in range(N_ACTIONS):
                if action_idx == current_action:
                    continue
                word_to_action[w] = action_idx
                acc = evaluate_accuracy(dataset, word_to_action, start_state)
                
                if acc > best_acc:
                    best_acc = acc
                    improved = True
                    current_action = action_idx
                else:
                    word_to_action[w] = current_action  # backtrack
                    
        print(f"Search Step {step} | Best Accuracy: {best_acc * 100:.2f}% | Elapsed: {time.time() - t0:.1f}s")
        
    print("\n--- Training Finished! ---")
    print(f"Final Accuracy: {best_acc * 100:.2f}% (Baseline random: {initial_acc * 100:.2f}%)")
    
    # Show a few sample trajectories
    print("\nSample Trajectories:")
    for i in range(min(3, len(dataset))):
        words, label = dataset[i]
        h_final = run_trajectory(words, word_to_action, start_state)
        print(f"  Words: {' '.join(words[:10])}...")
        print(f"  Target: {label:10s} | Collapsed State: {h_final}")

if __name__ == "__main__":
    main()
