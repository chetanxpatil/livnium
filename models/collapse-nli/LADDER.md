# The Collapse Ladder

One mechanism, stacked four times. Each rung learns a set of **wells**, freezes
them, and hands them up as the substrate for the rung above. Same force law
everywhere — only the wells change (letters → words):

```
h  <-  h  -  strength * (1 - cos(h, well)) * normalize(h - well)
```

A state is kept per position, so the trajectory can be read back (decode = nearest
well). No MLP, no GRU in the structure rungs — all knowledge lives in the wells.

## The chain

| Rung | Wells are… | Learns to… | File | Checkpoint | Result |
|------|-----------|-----------|------|-----------|--------|
| 1 | characters | write the **word** | `char_typer_symbols.py` | `char_typer_symbols.pt` | 100% per-symbol & exact-word (held-out) |
| 2 | words | write the **sentence** | `sentence_typer.py` | `sentence_typer.pt` | 100% per-word & exact-sentence (held-out) |
| 3 | words (char-scaffold) | grow **meaning** | `train_nli_meaning_forms_symbols.py` | `nli_meaning_forms_symbols.pt` | dev 66.83% (best, epoch 8) |
| 3′ | words (order-aware) | grow **meaning** | `train_nli_meaning_forms_sentence.py` | `nli_meaning_forms_sentence.pt` | order-aware variant |
| 4 | words (sentence-writer) | grow **meaning** | `train_nli_meaning_forms_sentence_typer.py` | `nli_meaning_forms_sentence_typer.pt` | meaning on the frozen writer |

## How the rungs are physically linked

The linkage is not conceptual — each file `torch.load`s the rung below it:

- **Rung 1 → 2.** `sentence_typer.py` builds word vectors; the word wells it learns
  sit on top of the character geometry from rung 1.
- **Rung 1 → 3.** `train_nli_meaning_forms_symbols.py` calls
  `build_char_scaffold()`, which loads `char_typer_symbols.pt` and places every
  word by the mean of its character wells (punctuation & digits included). That
  frozen scaffold is the starting position; a trainable residual (init 0) grows
  meaning around it.
- **Rung 2 → 4.** `train_nli_meaning_forms_sentence_typer.py` loads
  `sentence_typer.pt` and freezes its `word_anchors` **and** its collapse
  dynamics (`start`, `strength`). Meaning grows as a residual around those frozen
  word wells.

Break any lower rung and every rung above it changes — that's what makes it a
real ladder rather than four separate models.

## What each rung proves

- **Structure (rungs 1–2):** the collapse trajectory can carry and replay a
  sequence losslessly — first characters into a word, then words into a sentence.
  Both hit 100% on held-out data because the wells are per-token, not memorized
  sequences: any new arrangement of known tokens types back.
- **Meaning (rungs 3–4):** freeze the structure, grow a residual under
  supervision. Pure char geometry alone gives ~42% on SNLI (chance-ish, letters
  aren't meaning); letting meaning form around the frozen char scaffold lifts dev
  to ~66.8%. Rung 4 tests whether forming meaning around the richer *sentence*
  writer moves that number.

## Honest caveats

- Rungs 1–2 hit 100% because they keep a **state per position** (no compression).
  The NLI rungs pool to a single sentence vector, so some of that structural
  richness is squeezed out at the pool step — that's the current ceiling, not the
  geometry failing.
- The word wells in rung 2 cover the 20k most frequent words; rarer words share a
  single `<unk>` well and can't be typed/representation-resolved exactly. Raise
  `--max-vocab` for fuller coverage (costs memory in the readout).
- Rung-3 dev (66.83%) is a held-out slice of the train file — unseen in training,
  but selected on. For a fully independent number, score on `snli_1.0_test.jsonl`
  (`eval_nli_meaning_forms_symbols.py`).

## Helpers

- `score_sentence_typer.py` — load `sentence_typer.pt` and score it (batched) with
  no retraining.
- `eval_nli_meaning_forms_symbols.py` — score the rung-3 checkpoint on unseen data
  (real test set if present, else the reconstructed held-out split).
