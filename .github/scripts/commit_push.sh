#!/usr/bin/env bash
# Usage: commit_push.sh "<commit message>" path [path ...]
# Commits whichever of the given paths exist (missing ones are skipped instead of failing
# `git add` with exit 128) and pushes, rebasing on concurrent pushes.
set -u
msg="$1"; shift
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
paths=()
for p in "$@"; do [ -e "$p" ] && paths+=("$p"); done
if [ ${#paths[@]} -eq 0 ]; then echo "nothing to add"; exit 0; fi
find "${paths[@]}" -type f -size +90M -print -delete 2>/dev/null || true
git add -A -- "${paths[@]}"
git commit -q -m "$msg" || { echo "nothing to commit"; exit 0; }
for i in 1 2 3 4 5; do
  if git pull -q --rebase && git push -q; then echo "pushed"; exit 0; fi
  git rebase --abort 2>/dev/null || true
  sleep $((i*5))
done
echo "::error::push failed after 5 attempts"
exit 1
