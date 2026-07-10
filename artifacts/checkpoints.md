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
| `noun_collapse_pure.pt` (**v1**) | Wikipedia noun embeddings (pure collapse) | 104,704,445 B | `537707dfdd82a6caffacadfa683a88f32029e0bfbdcb8edb380a4f429953a4ea` | [🤗 chetanxpatil/noun-collapse](https://huggingface.co/chetanxpatil/noun-collapse) |
| `nli_epoch20.pt` | Supervised Collapse NLI v1 (`research/nli/supervised-collapse/`) | 52,614,341 B | `2ae8026dc25deaeb7a904b3980ed1fc6b95312874304d05b201bf645b796958d` | ⚠️ _upload pending — see below_ |
| `premise_from_hyp_align_53.pt` | SNLI premise generator (`research/generation/chat-brain/`) | 24,306,725 B | `a5ba5abcd140d80d8bbf19c7a1fdb5129bb733b57154d82fee49d2de8251001a` | ⚠️ _upload pending — see below_ |

**Pending uploads** — the two ⚠️ rows have no download URL yet. Until they are
uploaded, these checkpoints exist only on the author's machine. To publish them
as a GitHub Release and make this manifest fully verifiable:

```bash
gh release create checkpoints-v1 \
  research/nli/supervised-collapse/model_nli_v1/nli_epoch20.pt \
  research/generation/chat-brain/model/premise_from_hyp_align_53.pt \
  --title "Model checkpoints v1" \
  --notes "SHA-256 hashes in artifacts/checkpoints.md"
```

then replace the ⚠️ cells with
`https://github.com/chetanxpatil/livnium/releases/tag/checkpoints-v1`.

## Checkpoint versioning

**`noun_collapse_pure.pt` v1**: the published v1 checkpoint used the original
sampled-softmax implementation, *without* false-negative masking. Masking (a
sampled negative equal to the true target is now excluded with `-inf`) was
added to `research/embeddings/noun-collapse/noun_collapse_pure.py` afterward; a **v2 retrain is pending**.
Consequence: retraining with the current code will not exactly reproduce the
v1 checkpoint or its SimLex ρ = 0.362. The v1 *evaluation* is unaffected
(masking only changes training).

## Expected local paths

Scripts expect checkpoints at these paths (all gitignored):

```
research/generation/chat-brain/model/noun_collapse_pure.pt
research/generation/chat-brain/model/premise_from_hyp_align_53.pt
research/nli/supervised-collapse/model_nli_v1/nli_epoch20.pt
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
