# GitHub Export Guide

Updated: 2026-07-27

## Publication boundary

There are two deliberately different memory products:

- **Private forensic memory:** `/Users/chetanpatil/Desktop/LIVNIUM_MEMORY`
- **Public-safe package:** `/Users/chetanpatil/Desktop/test/LIVNIUM_MEMORY_GITHUB`

Only the public-safe package is a GitHub candidate. The private memory contains
machine paths, conversation provenance, repository topology, security notes,
audit probes, and detailed artifact locations needed for local recovery. It
must not be copied wholesale into a public repository.

## What the public package contains

- the durable mathematical and engineering core;
- project-family lineage and canonical-role rules;
- an evidence-bounded claim ledger;
- reusable and retired research ideas;
- family-level recovery coverage;
- fresh test evidence and its limits;
- archive hashes for identity without archive contents;
- one falsifiable next experiment;
- a privacy boundary and publication-status record; and
- a SHA-256 manifest for the files prepared for publication.

## What it excludes

- conversation exports, quoted chats, and personality material;
- usernames and absolute local paths;
- repository remotes and credential history;
- tokens, keys, cookies, environment files, and secrets;
- source datasets, checkpoints, caches, generated artifacts, and large models;
- machine-wide inventories and detailed source locations; and
- scripts that default to local private roots.

## Verified package state

The package was checked on 2026-07-27 for:

- absolute local paths and the local username;
- email addresses, URLs, repository remotes, and common credential patterns;
- symlinks, binary files, and files larger than 1 MiB;
- relative Markdown link resolution; and
- agreement between the public claims and the private claim/test ledgers.

No publish-blocking match was found. Matches for words such as “credentials,”
“tokens,” and `.env` occur only in the privacy policy and ignore rules. The
package has no Git history or remote, so history and remote inspection becomes
required after repository initialization.

The SHA-256 of `MANIFEST.sha256` itself is:

`20e701c62503e2cf947da191915ca9108c2c99f3f8dc09ce25d010a117636a99`

## Before the first push

1. Work from the public package directory only.
2. Read `PUBLICATION_STATUS.md`, `PRIVACY.md`, and `MANIFEST.sha256`.
3. Verify the manifest before adding or changing files.
4. Choose a reuse license intentionally. No license is selected yet.
5. Initialize Git only inside the public package.
6. Inspect the full staged diff and confirm no private material was added.
7. Inspect the configured remote before pushing.
8. Re-run privacy and secret scanning after every future addition.

No repository, commit, remote, or push was created during the recovery audit.
