#!/usr/bin/env bash

set -euo pipefail

memory_root="/Users/chetanpatil/Desktop/LIVNIUM_MEMORY"
inventory_root="${memory_root}/inventory"

mkdir -p "${inventory_root}"

roots=(
  "/Users/chetanpatil/Desktop/lets_clean_it/livnium"
  "/Users/chetanpatil/Desktop/test"
  "/Users/chetanpatil/Desktop/core"
  "/Users/chetanpatil/Desktop/livnium"
  "/Users/chetanpatil/Desktop/livnium-sacred"
  "/Users/chetanpatil/Desktop/livnium-sacred copy"
  "/Users/chetanpatil/Desktop/uantum"
)

roots_tmp="${inventory_root}/roots.tsv.tmp"
sources_tmp="${inventory_root}/source_files.tsv.tmp"
large_tmp="${inventory_root}/large_artifacts.tsv.tmp"
repos_tmp="${inventory_root}/git_repositories.tsv.tmp"

printf 'size_kib\tfiles\tsource_docs\tgit_branch\tgit_commit\tpath\n' > "${roots_tmp}"
printf 'sha256\tsize_bytes\tmodified_epoch\tpath\n' > "${sources_tmp}"
printf 'size_bytes\tmodified_epoch\tpath\n' > "${large_tmp}"
printf 'branch\tcommit\tremote\tpath\n' > "${repos_tmp}"

for root in "${roots[@]}"; do
  if [[ ! -d "${root}" ]]; then
    continue
  fi

  size_kib="$(du -sk "${root}" | awk '{print $1}')"
  file_count="$(find "${root}" -type f | wc -l | tr -d ' ')"
  source_count="$(
    find "${root}" \
      -type d \( -name .git -o -name node_modules -o -name .venv -o -name __pycache__ -o -name .dart_tool -o -name build \) -prune \
      -o -type f \( -name '*.py' -o -name '*.dart' -o -name '*.md' -o -name '*.txt' -o -name '*.html' -o -name '*.js' -o -name '*.toml' -o -name '*.yaml' -o -name '*.yml' -o -name '*.csv' \) -print \
      | wc -l | tr -d ' '
  )"

  branch="$(git -C "${root}" branch --show-current 2>/dev/null || true)"
  commit="$(git -C "${root}" rev-parse HEAD 2>/dev/null || true)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${size_kib}" "${file_count}" "${source_count}" "${branch}" "${commit}" "${root}" \
    >> "${roots_tmp}"

  find "${root}" \
    -type d \( -name .git -o -name node_modules -o -name .venv -o -name __pycache__ -o -name .dart_tool -o -name build \) -prune \
    -o -type f \( -name '*.py' -o -name '*.dart' -o -name '*.md' -o -name '*.txt' -o -name '*.html' -o -name '*.js' -o -name '*.toml' -o -name '*.yaml' -o -name '*.yml' -o -name '*.csv' \) -size -5M -print0 \
    | while IFS= read -r -d '' file; do
        hash_line="$(shasum -a 256 "${file}" 2>/dev/null || true)"
        hash="${hash_line%% *}"
        if [[ -z "${hash}" ]]; then
          continue
        fi
        size="$(stat -f '%z' "${file}" 2>/dev/null || true)"
        modified="$(stat -f '%m' "${file}" 2>/dev/null || true)"
        if [[ -z "${size}" || -z "${modified}" ]]; then
          continue
        fi
        printf '%s\t%s\t%s\t%s\n' "${hash}" "${size}" "${modified}" "${file}"
      done >> "${sources_tmp}" || true

  find "${root}" \
    -type d \( -name .git -o -name node_modules -o -name .venv -o -name __pycache__ -o -name .dart_tool -o -name build \) -prune \
    -o -type f -size +5M -print0 \
    | while IFS= read -r -d '' file; do
        size="$(stat -f '%z' "${file}" 2>/dev/null || true)"
        modified="$(stat -f '%m' "${file}" 2>/dev/null || true)"
        if [[ -z "${size}" || -z "${modified}" ]]; then
          continue
        fi
        printf '%s\t%s\t%s\n' "${size}" "${modified}" "${file}"
      done >> "${large_tmp}" || true
done

# Some dependency worktrees (notably ECW-BT's WikiExtractor copies) sit more
# than six levels below Desktop. Keep a generous finite bound so the registry
# sees them without walking unrelated mounted trees.
find /Users/chetanpatil/Desktop -maxdepth 14 -type d -name .git -prune -print0 \
  | while IFS= read -r -d '' git_dir; do
      repo="${git_dir%/.git}"
      branch="$(git -C "${repo}" branch --show-current 2>/dev/null || true)"
      commit="$(git -C "${repo}" rev-parse HEAD 2>/dev/null || true)"
      remote="$(git -C "${repo}" remote get-url origin 2>/dev/null || true)"
      # Never persist credentials or user-info embedded in an HTTPS remote.
      if [[ "${remote}" == http://*@* || "${remote}" == https://*@* ]]; then
        remote="HTTPS_REMOTE_REDACTED_USERINFO"
      fi
      printf '%s\t%s\t%s\t%s\n' "${branch}" "${commit}" "${remote}" "${repo}"
    done >> "${repos_tmp}" || true

mv "${roots_tmp}" "${inventory_root}/roots.tsv"
mv "${sources_tmp}" "${inventory_root}/source_files.tsv"
mv "${large_tmp}" "${inventory_root}/large_artifacts.tsv"
mv "${repos_tmp}" "${inventory_root}/git_repositories.tsv"

printf 'Inventory written to %s\n' "${inventory_root}"
