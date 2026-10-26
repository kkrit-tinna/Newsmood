#!/usr/bin/env bash
# Commit data/newsmood.db to the orphan `data` branch as newsmood.db (T4.4).
# Shared by ingest.yml and daily.yml. Works in a temporary worktree of
# origin/data so the main checkout is never switched. Run from the repo root
# after the git identity is configured.
#
# Usage: push_db.sh "<commit message>"
# "[skip ci]" is appended to the message.
set -euo pipefail

msg="${1:?usage: push_db.sh <commit message>}"
db="$PWD/data/newsmood.db"
[ -f "$db" ] || { echo "push_db: $db not found" >&2; exit 1; }

git fetch --depth=1 origin +refs/heads/data:refs/remotes/origin/data
wt="$(mktemp -d)"
trap 'git worktree remove --force "$wt" >/dev/null 2>&1 || true' EXIT
git worktree add --detach "$wt" origin/data

cp "$db" "$wt/newsmood.db"
cd "$wt"
git add newsmood.db
if git diff --cached --quiet; then
  echo "push_db: newsmood.db unchanged, nothing to push" >&2
  exit 0
fi
git commit -m "$msg [skip ci]"
# No retry or rebase: a binary DB can't be merged, and the shared concurrency
# group means no other workflow run pushes `data` meanwhile. A rejected push
# fails the job loudly instead of overwriting someone else's DB.
git push origin HEAD:refs/heads/data
