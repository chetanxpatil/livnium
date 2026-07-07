"""
scratch/test_batch_degradation.py — Verify if batching degrades representation quality

Runs a 3-epoch comparison:
1. Batch Size = 1 (pure online sequential updates)
2. Batch Size = 128 (standard batched updates)
"""

import sys
import os
import time
import random
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../experiment"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../chat"))

from pure_reply import (
    PureReplyBrain, load_wells_from_typer, read_pairs, shrink_vocab,
    DEFAULT_DATA, DEFAULT_TYPER_CKPT, DEFAULT_SEMANTIC_INIT,
    encode_ctx, encode_batch, CTX_WORDS, MAXLEN, semantic_init
)

def run_experiment(batch_size, device, train_msg, train_rep, dev_msg, dev_rep, model_args, minter_table):
    # Initialize fresh model and optimizer
    model = PureReplyBrain(*model_args).to(device)
    model.oov_wells = minter_table
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    
    t0 = time.time()
    for epoch in range(1, 3):
        model.train()
        perm = torch.randperm(train_msg.size(0))
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, train_msg.size(0), batch_size):
            idx = perm[i:i + batch_size]
            b_msg = train_msg[idx].to(device)
            b_rep = train_rep[idx].to(device)
            
            optimizer.zero_grad()
            loss = model.reply_nll(b_msg, b_rep)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
        # Eval
        model.eval()
        with torch.no_grad():
            dev_nll = model.reply_nll(dev_msg.to(device), dev_rep.to(device)).item()
            
    dt = time.time() - t0
    return epoch_loss / n_batches, dev_nll, dt

def main():
    device = torch.device("cpu")
    print("Loading data...")
    pairs = read_pairs(DEFAULT_DATA, max_lines=150)
    random.seed(42)
    random.shuffle(pairs)
    
    n_dev = 20
    train_pairs, dev_pairs = pairs[:-n_dev], pairs[-n_dev:]
    
    warm, stoi, itos, unk, eos, n_words, dim, extras = load_wells_from_typer(device, DEFAULT_TYPER_CKPT)
    warm, stoi, itos, unk, eos, n_words, minted = shrink_vocab(
        train_pairs, warm, stoi, itos, unk, eos, min_freq=2
    )
    
    # Encode context (CPU)
    from chat_reply import WordMinter
    minter = WordMinter(dim, n_words, device, DEFAULT_SEMANTIC_INIT)
    
    train_rep = encode_batch([r for _, r in train_pairs], stoi, unk, eos)
    train_msg = encode_ctx([m for m, _ in train_pairs], stoi, unk, eos, CTX_WORDS, minter=minter)
    dev_msg = encode_ctx([m for m, _ in dev_pairs], stoi, unk, eos, CTX_WORDS, minter=minter)
    dev_rep = encode_batch([r for _, r in dev_pairs], stoi, unk, eos)
    
    model_args = (n_words, dim, eos, warm)
    minter_table = minter.table()
    
    print("\n--- Running Batch Size = 1 (Online) ---")
    loss_1, dev_nll_1, time_1 = run_experiment(1, device, train_msg, train_rep, dev_msg, dev_rep, model_args, minter_table)
    print(f"BS=1   | Loss: {loss_1:.4f} | Dev NLL: {dev_nll_1:.4f} | Time: {time_1:.1f}s")
    
    print("\n--- Running Batch Size = 128 (Batched) ---")
    loss_128, dev_nll_128, time_128 = run_experiment(128, device, train_msg, train_rep, dev_msg, dev_rep, model_args, minter_table)
    print(f"BS=128 | Loss: {loss_128:.4f} | Dev NLL: {dev_nll_128:.4f} | Time: {time_128:.1f}s")
    
    print("\n==================================================")
    print("                  COMPASSION TABLE                ")
    print("==================================================")
    print(f"Batch Size 1   -> Dev NLL: {dev_nll_1:.4f} | Loss: {loss_1:.4f} | Time: {time_1:.1f}s")
    print(f"Batch Size 128 -> Dev NLL: {dev_nll_128:.4f} | Loss: {loss_128:.4f} | Time: {time_128:.1f}s")
    print("==================================================")
    
    nll_diff = dev_nll_128 - dev_nll_1
    if abs(nll_diff) < 0.05:
        print("Verdict: Batching does NOT degrade quality. Convergence is identical.")
    elif nll_diff > 0:
        print(f"Verdict: Batching degrades quality slightly (+{nll_diff:.4f} NLL).")
    else:
        print(f"Verdict: Batching improves quality slightly ({nll_diff:.4f} NLL).")

if __name__ == "__main__":
    main()
