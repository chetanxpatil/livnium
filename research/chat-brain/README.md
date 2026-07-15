# Chat Brain — grade B+

The active personal-chat research ladder. This folder is now only about the
char→word→conversation generator; the noun model and SNLI premise generator live
under `models/`.

## Mechanism

Every trained token is a distinct well. Reading a sequence produces an ordered
trajectory through those wells. The reply model keeps intermediate states, builds
a thought vector, and generates by selecting and collapsing toward output wells.
Unseen context words can be minted deterministically from spelling.

## Data boundary

`paths.py` is the single source of local paths:

- raw private export: repository-root `conversations.json` (gitignored),
- generated data: `research/chat-brain/data/` (gitignored),
- checkpoints: `research/chat-brain/model/` (gitignored),
- semantic initialization: `models/noun-collapse/model/`.

`prep_chat_context.canonical_turns()` follows only the displayed branch of each
conversation. Sessions never bleed into one another, and old context is trimmed
before recent turns.

## Ladder

| rung | file | purpose |
|---|---|---|
| characters | `char_typer_all.py` | reconstruct raw lines from character wells |
| words | `chat_typer.py` | reconstruct sentences from ordered word trajectories |
| live words | `chat_typer_live.py` | mint unseen word wells from spelling |
| conversation | `chat_reply.py` | read tagged multi-turn context and generate a reply |
| approximation | `fast_reader.py` | test a parallel approximation to sequential reading |

## Run

```bash
python3 research/chat-brain/prep_chat_context.py
python3 research/chat-brain/prep_chat_sentences.py
python3 research/chat-brain/chat_typer.py
python3 research/chat-brain/chat_reply.py
python3 research/chat-brain/chat_reply.py --chat
```

For a general-fluency stage before personal fine-tuning:

```bash
python3 research/chat-brain/prep_dailydialog.py
python3 research/chat-brain/chat_reply.py \
  --data research/chat-brain/data/dd_context.tsv \
  --extra-vocab research/chat-brain/data/chat_context.tsv \
  --ckpt research/chat-brain/model/chat_reply_general.pt
```

## Honest status

This is a promising research system, not a general chatbot. It can learn local
fluency and personal phrasing, but reasoning and robust label/constraint control
remain open problems. Promoted, measured results are documented separately in
`models/noun-collapse/` and `models/premise-generator/`.
