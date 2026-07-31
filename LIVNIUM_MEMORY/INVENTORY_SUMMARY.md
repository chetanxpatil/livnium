# Inventory Summary

Generated: 2026-07-26; audit coverage updated 2026-07-27

This is a content inventory, not a quality ranking. A file absent from the July
repository is protected for review; it is not automatically valuable or active.

## Coverage

- Livnium roots scanned: **7**
- Source/document rows: **17,403**
- Unique SHA-256 content hashes: **6,132**
- Hash groups with more than one copy: **2,249**
- Extra duplicate copies represented by those groups: **11,271**
- Large artifacts indexed by path, size, and time: **767**
- Git repositories in the generated broad registry: **27**
- Additional deep Git worktrees found during focused closure: **3** exact
  third-party WikiExtractor checkouts

The source manifest hashes files smaller than 5 MB with extensions used for
code, documentation, configuration, HTML, JavaScript, and CSV. Dependencies,
build folders, Git internals, caches, and large binary artifacts are not hashed.

## Root coverage

| Root | Source/docs hashed | Exact content also present in July canonical repo | Content absent from July canonical repo |
|---|---:|---:|---:|
| `lets_clean_it/livnium` | 418 | 418 | 0 |
| `test` | 10,850 | 55 | 10,795 |
| `core` | 503 | 3 | 500 |
| `livnium` | 180 | 0 | 180 |
| `livnium-sacred` | 75 | 0 | 75 |
| `livnium-sacred copy` | 5,319 | 0 | 5,319 |
| `uantum` | 58 | 0 | 58 |

These figures compare exact file bytes. A rewritten concept will count as absent
even when a later document discusses the same idea, which is why semantic review
is still necessary.

## Where the protected material is concentrated

Within `test`, source/docs absent from the July repository are concentrated in:

- `lab/`: 9,681 files
- `livnium.core-0.0.1/`: 371 files
- `_ORGANIZED/`: 119 exact-content-absent files
- `livnium-sacred/`: 91 files
- `livnium-sacred-v2/`: 67 files
- `nova-memory-main/`: 50 files
- `git-final/`: 40 files
- `nova/`: 32 files
- `livnium-public/`: 22 files
- `cube_embed/`: 21 files
- `livnium-core-clean/`: 20 files

This confirms that `lets_clean_it/livnium` is a curated current repository, not a
complete archive of all ideas.

## Focused arch-archive coverage

The broad manifest has now been followed by a focused audit of all three
1.8 GB `arch-archive` roots. Exact meaningful-subtree comparison establishes:

- `clean=noba=back/arch-archive` is the oldest self-contained mirror;
- workspace `arch-archive` is artifact-complete because it alone keeps the
  already audited K17/K4 Ramsey checkpoint;
- the `clean-nova-livnium` mirror omits the 136 meaningful base-Core files;
- shared `brain`, `core-c`, `language`, and `market-killer` content is exact
  across all three roots, and base Core is exact across the two copies that
  contain it.

The base Core, Core-C, market data/code, layer note, `important.md`, cache,
figures, and empty log stubs are now semantically incorporated in
`ARCH_ARCHIVE_ROOT_AUDIT.md`. The machine-readable broad-inventory counts above
were not regenerated, so their original generation date remains explicit.

## Final focused closure

The broad manifest is now complemented by:

- `NOVA_MISC_AUDIT.md`: all thirteen remaining organized scripts, their exact
  root-copy identity, Sacred evaluator leakage, Nova v1/v2 contracts, basin
  diagnostics, and observer redundancy;
- `REALCORE_LEGACY_SNAPSHOTS_AUDIT.md`: the February Python Core delta, 231-file
  older archive layer, embedded quantum releases, every named P2 snapshot,
  unique Livnium Crux release, and nine Git worktrees below `lab/infected`; and
- direct `_ORGANIZED` reconciliation: 203 meaningful regular files, 202 exact
  root copies, one generated index, and 27 of 28 resolving links.

The three WikiExtractor checkouts were missed because the original Git scan
stopped at depth six. `tools/build_inventory.sh` now searches to depth fourteen;
the TSV remains a dated 2026-07-26 snapshot until the expensive full inventory
is deliberately regenerated.

## Large artifacts

The manifest shows extensive duplication of datasets and model caches. Examples:

- one Wikipedia dump is approximately 25.8 GB;
- another Wikipedia dump is approximately 5.1 GB;
- a Rule30 dynamics artifact is approximately 3.5 GB;
- two pairwise-vector checkpoints are approximately 2.0 GB each;
- the SNLI training JSONL appears in many project copies.

No large artifact was deleted or moved. Later storage cleanup should hash large
files in focused batches before selecting archival copies.

## Machine-readable files

- `inventory/roots.tsv`
- `inventory/source_files.tsv`
- `inventory/large_artifacts.tsv`
- `inventory/git_repositories.tsv`

Regenerate them with `tools/build_inventory.sh`.

## Security boundary

The inventory does not intentionally store secrets. One unrelated local Git
remote was found with embedded HTTPS user-info during discovery. The generated
manifest was sanitized and the inventory script now replaces such remotes rather
than recording them. See `SECURITY_NOTES.md`.
