# Upload the noun-collapse model to the Hugging Face Hub

This folder is a ready-to-push HF model repo. One file is missing on purpose —
the checkpoint — because it is large and lives in `models/noun-collapse/model/`. Copy it in
first, then upload.

```
models/noun-collapse/huggingface/
├── README.md                    # model card (shown on the HF page)
├── config.json                  # architecture metadata
├── modeling_noun_collapse.py    # standalone loader (torch only)
└── noun_collapse_pure.pt        # <-- COPY THIS IN (see step 1), ~100 MB
```

## 1. Put the checkpoint in this folder

```bash
cp models/noun-collapse/model/noun_collapse_pure.pt \
  models/noun-collapse/huggingface/
```

## 2. One-time setup

```bash
pip install -U huggingface_hub
hf auth login    # token (write scope) from https://huggingface.co/settings/tokens
```

(The old `huggingface-cli` command is deprecated — the CLI is now `hf`.)

## 3. Upload

CLI (simplest):

```bash
# creates the repo if needed, then pushes the whole folder
hf upload chetanxpatil/noun-collapse ./models/noun-collapse/huggingface . --repo-type=model
```

Or Python:

```python
from huggingface_hub import HfApi, create_repo

repo_id = "chetanxpatil/noun-collapse"
create_repo(repo_id, repo_type="model", exist_ok=True)
HfApi().upload_folder(folder_path="models/noun-collapse/huggingface", repo_id=repo_id,
                      repo_type="model")
```

## 4. Verify

```bash
hf download chetanxpatil/noun-collapse --local-dir /tmp/nc
cd /tmp/nc && python3 modeling_noun_collapse.py noun_collapse_pure.pt
# should print the cat/physics/war/india neighbors
```

Notes
- Replace `chetanxpatil/noun-collapse` with whatever repo id you want.
- `.pt` files are tracked with Git LFS automatically by the HF uploader.
- Don't commit `noun_collapse_pure.pt` into the Livnium git repo — it's large;
  keep it in `models/noun-collapse/model/` (gitignored) and only copy it here at upload time.
