# Lab Nova-SNLI Physics and Label-Routing Audit

Audited: 2026-07-26

## Scope

Read-only source root:

`/Users/chetanpatil/Desktop/test/lab/nova-snli`

This audit traced the physics-embedding and early Nova-v3 SNLI artifacts that
precede the Sacred-v2 torque branch. It inspected source lineage, checkpoint
metadata, training/evaluation interfaces, embedding diagnostics, and complete
SNLI dev/test behavior under label-blind and label-routed collapse.

No source project, dataset, checkpoint, or archive was modified.

## Short conclusion

This root preserves the clearest recovered demonstration of the dynamic-basin
label shortcut.

- Its unique saved `snli_physics/best_model.pt` records a best dev accuracy of
  **33.33%**. Fresh deterministic, static, label-blind replay gives
  **32.99%** on SNLI test and predicts contradiction for 9,819 of 9,824
  examples.
- Feeding gold labels into its dynamic basin router raises the exact same
  checkpoint to **90.57% dev / 90.58% test**. Globally shuffled labels fall to
  **49.27% dev / 48.85% test**.
- The proper saved evaluator does use static collapse, so the stored 33% model
  is honestly selected near chance. But training routes every example through a
  ground-truth-label basin immediately before classification. The head learns a
  label-encoding shortcut that disappears at proper inference.
- This is direct recovered evidence for quarantining the later missing
  95.76%/96.07% result as provisionally leaked/unusable. It proves the failure
  mode existed and could create an impressive held-out number; it does not prove
  that this 90.58% checkpoint is the missing 96% artifact.
- The README's **74.4% test** claim is not supported by the only surviving
  checkpoint, log, or replay in this root.
- The full physics embedding artifact is a one-epoch 50,000×256 table. Its
  `physics_embeddings_final.pt` tensor is exactly equal to
  `physics_embed_epoch1.pt`.
- The embedding trainer optimizes only the word table. Its named collapse
  engine, including a randomly initialized learned update, is excluded from the
  optimizer. No seed or successful training log survives.
- The embedding table retains a weak local-context signal but fails all six
  runnable analogy examples in its own tiny benchmark. There is no matched
  standard skip-gram baseline proving a physics contribution.

Preservation role:
**canonical pre-Sacred physics-embedding and dynamic-label-shortcut evidence**.

## Root identity

- Approximate size: 812 MB.
- Primary branches: `physics_embed/` and `nova_v3/`.
- Six `.pt` paths represent five distinct hashes.
- The parent `/Users/chetanpatil/Desktop/test` is a Git repository, but
  `lab/nova-snli` itself is untracked and has no nested `.git`. Git history is
  therefore not provenance for this root.
- Source timestamps are primarily 2026-03-08; the test evaluator was modified
  on 2026-03-12.
- Raw SNLI train/dev/test hashes exactly match the other sacred replay roots.
- The unique SNLI checkpoint has SHA-256
  `914b033a58564f0954af2849f251f6badcbd3239a608f209c7505faac00aced7`
  and is the only file of its exact size found anywhere under Desktop.

Historical source inventory:

- 38 source/document rows and 38 distinct hashes;
- 20 hashes confined to this root at inventory time;
- 18 hashes copied elsewhere;
- nine direct source hashes shared with Sacred-v2, mostly backup quantum
  pretraining code and small core/utilities;
- the current physics-embedding trainer/evaluator and this SNLI
  training/evaluation version are distinct from Sacred-v2.

The root is a genuine predecessor, not merely a duplicate.

## Physics embedding artifact

### Checkpoints

| Artifact | SHA-256 | Meaning |
|---|---|---|
| full epoch 1 | `e0db4e00dad6b6c14e110d1b9acd994f7cd17c4ddd6a9360cd67e52b2be6eea3` | 50,000×256 state checkpoint; duplicate in `model_full_physics copy` |
| full final | `cfe0c7eac7e67e456de843f8e78d1ef5afcfe6d31ea5c95696b10b9e33883673` | Different container, but embedding tensor exactly equals full epoch 1 |
| smoke epoch 1 | `d3e864d0c18078aac0ab08e7676581ebe18b2e42d482abce494e6247a42a198f` | 12,716-word smoke state |
| smoke final | `f334b8d0574505d844c0eba8b8bcbb2095a3a084b9fee18cdf08376d8cdedadd` | 12,716-word smoke final |

The final full table has mean row norm 3.348 and includes 64 active
ALIGN basins plus 64 active CONTRAST basins. The final container saves the
embedding table, vocabulary, collapse state, and basin state, but no epoch,
training loss, seed, or complete command.

The only `train.log` is an argument-parser failure from the older
`train_quantum_embeddings.py`; it is not the successful run's log.

### What the trainer actually optimizes

The objective uses center/context pairs, random negatives, a target positive
cosine near 0.38, a negative margin, and a small norm penalty. It optionally
passes vectors through a collapse engine and updates dynamic basins.

However:

1. `Adam` receives only `PhysicsEmbeddingModel.parameters()`;
2. the collapse engine's anchors and two-layer learned update are not optimized;
3. no random seed is set;
4. dynamic basin centers update procedurally; and
5. the final file omits non-default collapse configuration except where defaults
   happen to reconstruct this run.

Thus “physics embeddings” accurately names the experimental objective and
routing wrapper, but not a jointly learned physical law.

### Focused quality diagnostics

On 200,000 held-out WikiText test context pairs, excluding padding, unknowns,
and identical tokens:

| Representation | Mean context-minus-random cosine | P(context > random) |
|---|---:|---:|
| Saved table | 0.00753 | 54.97% |
| Fixed row-shuffled control | 0.00435 | 51.03% |

This is a modest local-context signal, not evidence of a strong semantic
embedding.

The repository's eight hand-written analogy prompts produced:

- six runnable prompts, none returning the expected conventional answer;
- two prompts skipped because required lower-case city/country tokens were
  absent; and
- unrelated top answers such as `UNESCO`, `reissue`, `brainchild`, `Amiga`,
  `outpatient`, and `geologist`.

The benchmark is tiny and not a standard intrinsic evaluation, so its role is
diagnostic. It prevents promotion of the saved table as a demonstrated semantic
advance.

## Unique SNLI checkpoint

Path:

`nova_v3/model/snli_physics/best_model.pt`

Saved metadata:

- saved best dev accuracy: `0.3332655964`;
- global step: 549,288, corresponding to one full SNLI training epoch;
- configured epochs: 5;
- dynamic basins enabled;
- basin counts: 3 entailment, 1 neutral, 1 contradiction;
- about 13,054,609 model parameter elements;
- file size: 155,352,757 bytes because optimizer state and duplicated encoder
  material are also stored.

Parameter allocation:

| Component | Elements | Role |
|---|---:|---|
| Main SNLI collapse | 768 | Three 256-dimensional static anchors; no learned update |
| Encoder | 12,981,761 | Trainable 50,000×256 embeddings, gravity probe, and saved nested embedding-collapse state |
| Head | 72,080 | Direction/radius feature MLP |

The nested physics-embedding collapse engine is present inside the encoder state,
but `encode_sentence()` only applies the embedding table and gravity pooling. It
does not invoke that nested collapse. The main SNLI collapse contains no learned
MLP despite README text saying “learned update + physics force.”

The SNLI fine-tuned embedding table remains close to the one-epoch pretrained
table (mean matching-row cosine 0.9649) but is not frozen.

## Complete deterministic replay

Encoder noise was disabled, all examples were evaluated, and the actual class
order entailment, contradiction, neutral was used.

| Mode | Dev | Test | Behavior |
|---|---:|---:|---|
| Static label-blind, saved anchors | 33.34% | — | Dev predictions: 9,839 contradiction, 3 entailment |
| Static label-blind, basin-derived anchors | 33.34% | **32.99%** | Test predictions: 9,819 contradiction, 5 entailment |
| No main collapse | 32.87% | 32.77% | Predicts neutral for every example |
| Dynamic routing with gold labels | **90.57%** | **90.58%** | Direct label-conditioned shortcut |
| Dynamic routing with globally shuffled labels | 49.27% | 48.85% | Shortcut broken; substantial routing dependence remains |

Gold-routed test confusion matrix, with rows/columns in entailment,
contradiction, neutral order:

```text
[[3160, 201,   7],
 [ 712,2524,   1],
 [   3,   1,3215]]
```

Static test confusion:

```text
[[   4,3364,0],
 [   0,3237,0],
 [   1,3218,0]]
```

The scripts print confusion headers as entailment, neutral, contradiction even
though the dataset mapping and logits are entailment, contradiction, neutral.
This is a reporting bug.

### Interpretation

Using labels to organize training geometry is not automatically invalid. Here,
however, the router takes the exact target label for every example and selects a
label-specific basin immediately before the classifier. The classifier can read
the injected label geometry rather than learn NLI from premise and hypothesis.

The saved developer recognized the evaluation leak and switched dev/test to
static collapse. That correct evaluation exposes the train/inference mismatch:
the model is chance-level without the label shortcut.

There are no assertion-based tests under this root; `pytest` collects zero
tests. The available `test_*.py` files are smoke or diagnostic scripts.

## README and evidence boundary

The README claims:

- 74.4% test accuracy;
- 52.3 MB model size;
- 7,800+ CPU pairs/second;
- about 28 minutes of CPU-only training; and
- a frozen physics layer with a learned update.

For this root:

- the only SNLI checkpoint replays at 32.99% proper test accuracy;
- the complete checkpoint is 155.4 MB, while the separate embedding file is
  52.7 MB;
- no timing or completed training log supports throughput or duration;
- the encoder table is fine-tuned, not frozen;
- the main SNLI collapse has no learned update; and
- the embedding collapse update is randomly initialized, excluded from its
  optimizer, and not called during SNLI sentence encoding.

Preserve the README as historical intent, not as benchmark evidence.

## Relationship to the missing 96% result

This root does not contain 95.76%, 96.07%, or the named missing checkpoint.
Nevertheless, it establishes the exact ancestral failure mode:

1. dynamic basins accept ground-truth labels;
2. routed state becomes highly label-decodable;
3. correct label routing inflates a chance-level model to 90.58% test; and
4. shuffled or static routing destroys the result.

This makes the provisional leaked/unusable classification of the later 96%
historical result materially stronger. The link is a lineage inference, not
artifact identity.

## Preservation and continuation decision

- Preserve the whole root and its five distinct checkpoint hashes.
- Treat the physics embedding as a one-epoch exploratory artifact with a modest
  context signal, not a verified semantic advance.
- Treat `snli_physics/best_model.pt` as a measured negative/shortcut artifact:
  32.99% proper test versus 90.58% gold-routed test.
- Retire the root's 74.4% README benchmark and unsupported speed/training claims.
- Preserve the corrected static evaluator as evidence that the leak was noticed.
- Do not revive label-routed classification. If label-conditioned basins are
  studied again, the target label must never enter the state seen by the
  classifier or any held-out inference path.
- The source/artifact continuation at
  `test/lab/infected/Archive/nova_v3` and adjacent `quantum_embed` is now
  incorporated in `INFECTED_ARCHIVE_NOVA_AUDIT.md`. Continue with the nested
  eight-generation `NLI-ALL` archive recorded there.
