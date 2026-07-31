# Security Notes

Updated: 2026-07-27

## Embedded Git credential

During Desktop Git discovery, one unrelated Flutter backup repository had an
HTTPS remote containing embedded user-info that appeared credential-like.

Actions taken:

- The value was not copied into the durable memory documents.
- The generated Git manifest was sanitized.
- `tools/build_inventory.sh` now replaces any HTTPS remote containing user-info
  with a fixed redacted marker.

Remaining user action:

- Rotate or revoke that Bitbucket credential.
- Replace the affected local remote with a credential-free URL and use the
  system credential manager instead.

The affected repository is the backup copy under:

`/Users/chetanpatil/Desktop/flutter/backup/retail_flutter`

Do not paste the old remote URL into chat or documentation.

## GitHub publication boundary

Do not publish `/Users/chetanpatil/Desktop/LIVNIUM_MEMORY` directly. It
intentionally contains:

- absolute machine paths and a local username;
- private conversation chronology and personality/context files;
- unrelated repository metadata;
- security-remediation history;
- full forensic manifests of local projects and artifacts; and
- probes whose defaults point at local source roots.

The public package is generated separately at:

`/Users/chetanpatil/Desktop/test/LIVNIUM_MEMORY_GITHUB`

Its rules are:

1. no conversation export or personality profile;
2. no absolute local paths or private remotes;
3. no credentials, tokens, cookies, environment files, checkpoints, datasets,
   or local artifact manifests;
4. no claim stronger than the private evidence ledger;
5. only public-safe summaries and archive hashes; and
6. run a secret/path/privacy scan before any commit or push.

See `GITHUB_EXPORT_GUIDE.md`. No repository was created or pushed during the
recovery audit.
