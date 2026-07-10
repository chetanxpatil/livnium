# Checkpoint manifest

Trained checkpoints are **not** stored in git (they bloat the repo and its
history). They are published externally; this file is the single source of
truth for where each checkpoint lives and how to verify it.

Verify any download with:

```bash
shasum -a 256 <file>     # macOS
sha256sum <file>         # Linux
```

The hash must match this table exactly.

| Checkpoint | Model | Size | SHA-256 | Where |
|---|---|---:|---|---|
| `noun_collapse_pure.pt` | Wikipedia noun embeddings (pure collapse) | 104,704,445 B | `537707dfdd82a6caffacadfa683a88f32029e0bfbdcb8edb380a4f429953a4ea` | [🤗 chetanxpatil/noun-collapse](https://huggingface.co/chetanxpatil/noun-collapse) |
| `nli_epoch20.pt` | Supervised Collapse NLI v1 (`collapse_retrain/`) | 52,614,341 B | `2ae8026dc25deaeb7a904b3980ed1fc6b95312874304d05b201bf645b796958d` | _upload pending — GitHub Release `checkpoints-v1`_ |
| `premise_from_hyp_align_53.pt` | SNLI premise generator (`chat/`) | 24,306,725 B | `a5ba5abcd140d80d8bbf19c7a1fdb5129bb733b57154d82fee49d2de8251001a` | _upload pending — GitHub Release `checkpoints-v1`_ |

## Expected local paths

Scripts expect checkpoints at these paths (all gitignored):

```
chat/model/noun_collapse_pure.pt
chat/model/premise_from_hyp_align_53.pt
collapse_retrain/model_nli_v1/nli_epoch20.pt
```

## Publishing a new checkpoint

1. Upload to Hugging Face (preferred for models with cards) or attach to a
   GitHub Release (fine for experiment checkpoints).
2. Add a row here: filename, what it is, byte size, SHA-256, URL.
3. Never `git add` a `.pt`/`.npz` — the `.gitignore` blocks them; keep it so.

## Note on history size

Removing the two tracked checkpoints shrinks future clones, but ~76 MB of
already-committed checkpoint blobs remain in git history (`.git` ≈ 213 MB).
To reclaim it, after the checkpoints are uploaded and this manifest has their
final URLs, rewrite history once:

```bash
git filter-repo --strip-blobs-bigger-than 10M
git push --force-with-lease
```

(Coordinate before force-pushing — every clone must re-clone after a rewrite.)
