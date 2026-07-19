# Legacy chat preprocessing

`prep_chat_pairs.py` reads an old flattened conversation export. It was replaced
by the session-aware canonical-path pipeline in `research/chat-brain/` because the
flattened format loses branch and conversation boundaries.

The script and its local generated pairs are retained only for historical runs.
`link_chat_data.py` is the superseded helper that copied/symlinked active chat
data into an experiment directory; current experiments resolve the canonical
chat-data path directly.
